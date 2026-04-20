"""Extruded building meshes from OSM footprints."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import shapely
import shapely.geometry as sg
from mapbox_earcut import triangulate_float64

from osm3denv.fetch.osm import OSMData, OSMRelation, OSMWay
from osm3denv.frame import Frame
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

_NUM_RE = re.compile(r"([-+]?\d*\.?\d+)")


def parse_height(s: str | None) -> float | None:
    if not s:
        return None
    m = _NUM_RE.search(s)
    return float(m.group(1)) if m else None


def resolve_height(tags: dict[str, str]) -> float:
    h = parse_height(tags.get("height"))
    if h is not None and h > 0:
        return h
    levels = parse_height(tags.get("building:levels"))
    if levels is not None and levels > 0:
        return levels * 3.0
    return 6.0


@dataclass
class BuildingsMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    count: int


def _ring_to_enu(ring_ll, frame: Frame) -> np.ndarray:
    lon = np.asarray([p[0] for p in ring_ll], dtype=np.float64)
    lat = np.asarray([p[1] for p in ring_ll], dtype=np.float64)
    east, north = frame.to_enu(lon, lat)
    return np.stack([east, north], axis=-1)


def _is_closed(ring: np.ndarray, tol: float = 0.5) -> bool:
    return len(ring) >= 3 and np.linalg.norm(ring[0] - ring[-1]) <= tol


def _polygon_from_way(way: OSMWay, frame: Frame) -> sg.Polygon | None:
    ring = _ring_to_enu(way.geometry, frame)
    if not _is_closed(ring):
        return None
    try:
        poly = sg.Polygon(ring[:-1])
        fixed = shapely.make_valid(poly.buffer(0))
    except Exception:  # noqa: BLE001
        return None
    return _take_polygon(fixed)


def _polygon_from_relation(rel: OSMRelation, frame: Frame) -> sg.Polygon | None:
    outers = [r for (role, r) in rel.rings if role == "outer"]
    inners = [r for (role, r) in rel.rings if role == "inner"]
    if not outers:
        return None
    # v1 simplification: use the first closed outer as the polygon; any inner fully
    # contained inside it becomes a hole.
    for outer_ll in outers:
        outer = _ring_to_enu(outer_ll, frame)
        if not _is_closed(outer):
            continue
        try:
            base = sg.Polygon(outer[:-1])
            holes = []
            for inner_ll in inners:
                inner = _ring_to_enu(inner_ll, frame)
                if _is_closed(inner):
                    h = sg.Polygon(inner[:-1])
                    if base.contains(h):
                        holes.append(list(h.exterior.coords)[:-1])
            base = sg.Polygon(list(base.exterior.coords)[:-1], holes)
            fixed = shapely.make_valid(base.buffer(0))
        except Exception:  # noqa: BLE001
            continue
        poly = _take_polygon(fixed)
        if poly is not None:
            return poly
    return None


def _take_polygon(geom) -> sg.Polygon | None:
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area >= 1.0 else None
    if geom.geom_type == "MultiPolygon":
        best = max(geom.geoms, key=lambda g: g.area)
        return best if best.area >= 1.0 else None
    return None


def _extrude(poly: sg.Polygon, height_m: float, base_y: float):
    outer = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
    inners = [np.asarray(h.coords, dtype=np.float64)[:-1] for h in poly.interiors]

    rings = [outer] + inners
    flat2d = np.vstack(rings)
    ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        tri = triangulate_float64(flat2d, ring_ends).reshape(-1, 3)
    except Exception as exc:  # noqa: BLE001
        log.debug("earcut failed: %s", exc)
        return None

    n = len(flat2d)
    roof = np.stack([flat2d[:, 0], np.full(n, base_y + height_m),
                     -flat2d[:, 1]], axis=-1).astype(np.float32)
    floor = np.stack([flat2d[:, 0], np.full(n, base_y),
                      -flat2d[:, 1]], axis=-1).astype(np.float32)

    roof_norms = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)
    floor_norms = np.tile([0.0, -1.0, 0.0], (n, 1)).astype(np.float32)
    roof_idx = tri.astype(np.uint32)
    floor_idx = (tri[:, [0, 2, 1]] + n).astype(np.uint32)

    wall_verts: list[tuple[float, float, float]] = []
    wall_norms: list[tuple[float, float, float]] = []
    wall_idx: list[tuple[int, int, int]] = []
    base_offset = 2 * n
    for ring in rings:
        m = len(ring)
        for i in range(m):
            a = ring[i]
            b = ring[(i + 1) % m]
            de = b[0] - a[0]
            dn = b[1] - a[1]
            length = float(np.hypot(de, dn))
            if length < 1e-6:
                continue
            # Outward normal in ENU for a CCW polygon: (dn, -de). Map to Ogre
            # (x=east, y=up, z=-north): (dn, 0, -(-de)) = (dn, 0, de).
            normal = (dn / length, 0.0, de / length)
            a_b = (float(a[0]), base_y, float(-a[1]))
            b_b = (float(b[0]), base_y, float(-b[1]))
            b_t = (float(b[0]), base_y + height_m, float(-b[1]))
            a_t = (float(a[0]), base_y + height_m, float(-a[1]))
            v0 = base_offset + len(wall_verts)
            wall_verts.extend([a_b, b_b, b_t, a_t])
            wall_norms.extend([normal] * 4)
            wall_idx.extend([(v0, v0 + 1, v0 + 2), (v0, v0 + 2, v0 + 3)])

    if wall_verts:
        wv = np.asarray(wall_verts, dtype=np.float32)
        wn = np.asarray(wall_norms, dtype=np.float32)
        wi = np.asarray(wall_idx, dtype=np.uint32).reshape(-1)
    else:
        wv = np.zeros((0, 3), dtype=np.float32)
        wn = np.zeros((0, 3), dtype=np.float32)
        wi = np.zeros((0,), dtype=np.uint32)

    vertices = np.concatenate([roof, floor, wv], axis=0)
    normals = np.concatenate([roof_norms, floor_norms, wn], axis=0)
    indices = np.concatenate([roof_idx.ravel(), floor_idx.ravel(), wi], axis=0)
    return vertices, normals, indices


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler) -> BuildingsMesh:
    all_v: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    index_offset = 0
    count = 0

    def add(tuple_):
        nonlocal index_offset, count
        v, n, i = tuple_
        all_v.append(v)
        all_n.append(n)
        all_i.append(i + index_offset)
        index_offset += len(v)
        count += 1

    for w in osm.filter_ways(lambda t: "building" in t or "building:part" in t):
        poly = _polygon_from_way(w, frame)
        if poly is None:
            continue
        h = resolve_height(w.tags)
        cx, cy = poly.centroid.x, poly.centroid.y
        base_y = float(sampler.height_at(cx, cy))
        ext = _extrude(poly, h, base_y)
        if ext is not None:
            add(ext)

    for r in osm.filter_relations(lambda t: "building" in t):
        poly = _polygon_from_relation(r, frame)
        if poly is None:
            continue
        h = resolve_height(r.tags)
        cx, cy = poly.centroid.x, poly.centroid.y
        base_y = float(sampler.height_at(cx, cy))
        ext = _extrude(poly, h, base_y)
        if ext is not None:
            add(ext)

    if not all_v:
        empty = np.zeros((0, 3), dtype=np.float32)
        return BuildingsMesh(empty, empty, np.zeros((0,), dtype=np.uint32), 0)

    return BuildingsMesh(
        vertices=np.concatenate(all_v, axis=0),
        normals=np.concatenate(all_n, axis=0),
        indices=np.concatenate(all_i, axis=0),
        count=count,
    )
