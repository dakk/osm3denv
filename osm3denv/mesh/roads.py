"""Road and railway ribbons draped on the terrain.

Each OSM way is classified into one of five surface kinds (asphalt_major,
asphalt_minor, paved, dirt, rail) so the renderer can assign a different
material per kind and procedurally draw lane markings where appropriate.

Unlike the old implementation, ribbons are constructed parametrically (left
edge / right edge per centreline sample) instead of by buffering. The UV that
comes out is *road-local*:

    uv.x = lateral position in metres from the centreline  (negative = left)
    uv.y = cumulative distance along the road in metres

This lets the fragment shader paint lane markings at known offsets (e.g. u=0
for the centreline, v mod 12 for dashes) without needing the road's width
passed as a separate attribute.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely.geometry as sg
import shapely.vectorized as sv
from shapely.ops import unary_union

from osm3denv.fetch.osm import OSMData, OSMWay
from osm3denv.frame import Frame
from osm3denv.mesh.buildings import parse_height
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

_HIGHWAY_WIDTH = {
    "motorway": 12.0, "motorway_link": 8.0,
    "trunk": 10.0, "trunk_link": 7.0,
    "primary": 8.0, "primary_link": 6.0,
    "secondary": 6.0, "secondary_link": 5.0,
    "tertiary": 5.5, "tertiary_link": 4.5,
    "residential": 5.0, "unclassified": 5.0,
    "living_street": 4.0, "service": 3.0,
    "pedestrian": 4.0, "footway": 1.5, "path": 1.5, "cycleway": 2.0,
    "track": 3.0, "steps": 1.5, "bridleway": 2.0,
}
_RAIL_WIDTH = 3.0

# (kind priority ordered) — first match wins.
_KIND_MAJOR = {"motorway", "trunk", "primary",
               "motorway_link", "trunk_link", "primary_link"}
_KIND_MINOR = {"secondary", "tertiary", "residential", "unclassified",
               "service", "living_street",
               "secondary_link", "tertiary_link"}
_KIND_PAVED = {"footway", "pedestrian", "steps", "cycleway"}
_KIND_DIRT  = {"path", "track", "bridleway"}


@dataclass
class RoadsMesh:
    kind: str                  # "asphalt_major" | "asphalt_minor" | "paved" | "dirt" | "rail"
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    count: int


def _classify(tags: dict[str, str]) -> str:
    if "railway" in tags:
        return "rail"
    hw = tags.get("highway", "")
    if hw in _KIND_MAJOR:
        return "asphalt_major"
    if hw in _KIND_MINOR:
        return "asphalt_minor"
    if hw in _KIND_PAVED:
        return "paved"
    if hw in _KIND_DIRT:
        return "dirt"
    return "asphalt_minor"


def _way_width(way: OSMWay) -> float:
    w = parse_height(way.tags.get("width"))
    if w is not None and w > 0:
        return w
    if "railway" in way.tags:
        return _RAIL_WIDTH
    hw = way.tags.get("highway")
    if hw in _HIGHWAY_WIDTH:
        return _HIGHWAY_WIDTH[hw]
    return 4.0


def _way_to_coords(way: OSMWay, frame: Frame) -> np.ndarray | None:
    lon = np.asarray([p[0] for p in way.geometry], dtype=np.float64)
    lat = np.asarray([p[1] for p in way.geometry], dtype=np.float64)
    east, north = frame.to_enu(lon, lat)
    coords = np.stack([east, north], axis=-1)
    if len(coords) < 2:
        return None
    # Drop near-duplicates.
    d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    keep = np.concatenate([[True], d > 1e-3])
    coords = coords[keep]
    if len(coords) < 2:
        return None
    total = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    return coords if total > 0.5 else None


def _densify(coords: np.ndarray, max_step: float) -> np.ndarray:
    out = [coords[0]]
    for i in range(len(coords) - 1):
        a, b = coords[i], coords[i + 1]
        d = float(np.linalg.norm(b - a))
        if d > max_step:
            n = int(np.ceil(d / max_step))
            for k in range(1, n):
                t = k / n
                out.append(a + (b - a) * t)
        out.append(b)
    return np.asarray(out, dtype=np.float64)


def _build_ribbon(coords: np.ndarray, width: float,
                  sampler: TerrainSampler, *,
                  offset: float = 0.0, flat_y: float | None = None,
                  max_step: float = 1.5, lateral_offset: float = 0.0):
    """Construct a triangle-strip ribbon along the centreline ``coords``.

    ``lateral_offset`` shifts the ribbon perpendicular to the road direction
    (positive = to the left of the direction of travel). Used to place
    sidewalks beside the asphalt without changing the main-road ribbon.

    Returns (vertices, normals, indices, uvs) or None if degenerate.
    """
    dens = _densify(coords, max_step)
    n = len(dens)
    if n < 2:
        return None

    # Per-vertex tangent: central differences in the interior, one-sided at ends.
    tangents = np.zeros_like(dens)
    tangents[1:-1] = dens[2:] - dens[:-2]
    tangents[0] = dens[1] - dens[0]
    tangents[-1] = dens[-1] - dens[-2]
    tlen = np.linalg.norm(tangents, axis=1, keepdims=True)
    tlen = np.where(tlen < 1e-6, 1.0, tlen)
    tangents = tangents / tlen
    # Perpendicular in the East/North plane (rotate 90° CCW).
    perp = np.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)

    centre = dens + perp * lateral_offset
    half_w = width * 0.5
    left  = centre + perp * half_w
    right = centre - perp * half_w

    # Interleave left_i, right_i → 2n vertices.
    verts_2d = np.empty((2 * n, 2), dtype=np.float64)
    verts_2d[0::2] = left
    verts_2d[1::2] = right

    # Y: terrain-draped (per-vertex) or flat (bridges).
    if flat_y is None:
        ys = sampler.height_at(verts_2d[:, 0], verts_2d[:, 1]) + offset
    else:
        ys = np.full(2 * n, flat_y + offset, dtype=np.float32)
    ys = ys.astype(np.float32)

    # Ogre frame: (east, y, -north).
    vertices = np.stack([verts_2d[:, 0].astype(np.float32),
                         ys,
                         (-verts_2d[:, 1]).astype(np.float32)], axis=-1)

    # Cumulative along-road distance at each centreline sample.
    seg = np.linalg.norm(np.diff(dens, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)

    # UVs: u in metres from centreline, v in metres along road.
    uvs = np.empty((2 * n, 2), dtype=np.float32)
    uvs[0::2, 0] = -half_w
    uvs[1::2, 0] = +half_w
    uvs[0::2, 1] = cum
    uvs[1::2, 1] = cum

    normals = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32),
                      (2 * n, 1))

    # Two triangles per segment: (left_i, right_i, left_{i+1}) and
    # (right_i, right_{i+1}, left_{i+1}), wound CCW when viewed from above.
    tri = np.empty(((n - 1) * 2, 3), dtype=np.uint32)
    i = np.arange(n - 1, dtype=np.uint32)
    i0 = 2 * i; i1 = 2 * i + 1; i2 = 2 * i + 2; i3 = 2 * i + 3
    tri[0::2, 0] = i0; tri[0::2, 1] = i1; tri[0::2, 2] = i2
    tri[1::2, 0] = i1; tri[1::2, 1] = i3; tri[1::2, 2] = i2
    indices = tri.ravel()

    return vertices, normals, indices, uvs


def _road_union(candidates: list[OSMWay], frame: Frame):
    """Union of every (non-tunnel) highway's tarmac footprint.

    Used to clip sidewalk ribbons so they don't run across a crossing road's
    surface. Railways and footpaths are excluded because we don't build
    sidewalks for them (and their surfaces shouldn't clip anything either).
    """
    polys = []
    for way in candidates:
        if way.tags.get("tunnel") == "yes":
            continue
        if "railway" in way.tags:
            continue
        hw = way.tags.get("highway")
        if hw in _KIND_PAVED or hw in _KIND_DIRT:
            continue
        coords = _way_to_coords(way, frame)
        if coords is None:
            continue
        width = _way_width(way)
        try:
            line = sg.LineString(coords)
            polys.append(line.buffer(width * 0.5,
                                     cap_style="flat", join_style="mitre"))
        except Exception:  # noqa: BLE001
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:  # noqa: BLE001
        return None


def _clip_ribbon_triangles(ribbon, road_union):
    """Drop triangles whose centroid lies inside ``road_union``.

    Applied to sidewalk ribbons so a sidewalk can't cross a perpendicular
    road's tarmac at a junction. Vertices left dangling are harmless — Ogre
    only renders triangles via the index buffer.
    """
    if road_union is None:
        return ribbon
    vertices, normals, indices, uvs = ribbon
    tri = indices.reshape(-1, 3)
    # Centroids in ENU: east = vx, north = -vz (Ogre z is flipped).
    cx = (vertices[tri[:, 0], 0]
          + vertices[tri[:, 1], 0]
          + vertices[tri[:, 2], 0]) / 3.0
    cn = -(vertices[tri[:, 0], 2]
           + vertices[tri[:, 1], 2]
           + vertices[tri[:, 2], 2]) / 3.0
    inside = sv.contains(road_union,
                         cx.astype(np.float64),
                         cn.astype(np.float64))
    keep = ~inside
    if keep.all():
        return ribbon
    new_indices = tri[keep].ravel().astype(np.uint32)
    if len(new_indices) == 0:
        return None
    return vertices, normals, new_indices, uvs


def build(osm: OSMData, frame: Frame,
          sampler: TerrainSampler) -> list[RoadsMesh]:
    """Build one RoadsMesh per road kind encountered."""
    buckets: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    candidates = osm.filter_ways(lambda t: "highway" in t or "railway" in t)
    road_union = _road_union(candidates, frame)

    for way in candidates:
        if way.tags.get("tunnel") == "yes":
            continue
        coords = _way_to_coords(way, frame)
        if coords is None:
            continue

        kind = _classify(way.tags)
        width = _way_width(way)

        flat_y: float | None = None
        if way.tags.get("bridge") == "yes":
            layer = parse_height(way.tags.get("layer")) or 1.0
            lift = max(3.0, 5.0 * float(layer))
            ys = [float(sampler.height_at(x, y)) for x, y in coords]
            flat_y = max(ys) + lift
            res = _build_ribbon(coords, width, sampler,
                                offset=0.0, flat_y=flat_y, max_step=1.5)
        else:
            res = _build_ribbon(coords, width, sampler,
                                offset=0.40, max_step=1.5)
        if res is None:
            continue
        buckets.setdefault(kind, []).append(res)

        # Sidewalks alongside asphalt roads only (and only when on the
        # ground — elevated bridges don't get side strips). Two 1.5 m-wide
        # paving ribbons, centred 0.75 m outside each kerb, sitting 5 cm
        # higher than the tarmac so there's a visible step.
        if (kind in ("asphalt_major", "asphalt_minor")
                and not way.tags.get("bridge") == "yes"):
            side_w = 1.5
            lat = width * 0.5 + side_w * 0.5
            for sign in (+1.0, -1.0):
                side = _build_ribbon(coords, side_w, sampler,
                                     offset=0.45, max_step=1.5,
                                     lateral_offset=sign * lat)
                if side is None:
                    continue
                side = _clip_ribbon_triangles(side, road_union)
                if side is not None:
                    buckets.setdefault("sidewalk", []).append(side)

    out: list[RoadsMesh] = []
    for kind, parts in buckets.items():
        all_v = []; all_n = []; all_i = []; all_u = []
        off = 0
        count = 0
        for v, n, i, u in parts:
            all_v.append(v); all_n.append(n)
            all_i.append(i + off)
            all_u.append(u)
            off += len(v)
            count += 1
        vertices = np.concatenate(all_v, axis=0)
        out.append(RoadsMesh(
            kind=kind,
            vertices=vertices,
            normals=np.concatenate(all_n, axis=0),
            indices=np.concatenate(all_i, axis=0),
            uvs=np.concatenate(all_u, axis=0),
            count=count,
        ))
    return out
