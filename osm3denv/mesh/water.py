"""Water features: areas draped as flat polygons, streams as narrow ribbons."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely
import shapely.geometry as sg
import shapely.ops

from osm3denv.fetch.osm import OSMData, OSMRelation, OSMWay
from osm3denv.frame import Frame
from osm3denv.mesh.buildings import parse_height
from osm3denv.mesh.drape import drape
from osm3denv.mesh.sample import TerrainSampler
from mapbox_earcut import triangulate_float64

log = logging.getLogger(__name__)


@dataclass
class WaterMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    count: int


def _ring_to_enu(ring_ll, frame: Frame) -> np.ndarray:
    lon = np.asarray([p[0] for p in ring_ll], dtype=np.float64)
    lat = np.asarray([p[1] for p in ring_ll], dtype=np.float64)
    east, north = frame.to_enu(lon, lat)
    return np.stack([east, north], axis=-1)


def _polygon_from_way(way: OSMWay, frame: Frame) -> sg.Polygon | None:
    ring = _ring_to_enu(way.geometry, frame)
    if len(ring) < 4 or np.linalg.norm(ring[0] - ring[-1]) > 1.0:
        return None
    try:
        p = sg.Polygon(ring[:-1])
        p = shapely.make_valid(p.buffer(0))
    except Exception:  # noqa: BLE001
        return None
    return p if not p.is_empty and p.area >= 1.0 and p.geom_type == "Polygon" else None


def _polygons_from_relation(rel: OSMRelation, frame: Frame) -> list[sg.Polygon]:
    outers = [_ring_to_enu(r, frame) for (role, r) in rel.rings if role == "outer"]
    inners = [_ring_to_enu(r, frame) for (role, r) in rel.rings if role == "inner"]
    polys: list[sg.Polygon] = []
    for outer in outers:
        if len(outer) < 4 or np.linalg.norm(outer[0] - outer[-1]) > 1.0:
            continue
        holes = []
        base = sg.Polygon(outer[:-1])
        for inner in inners:
            if len(inner) < 4 or np.linalg.norm(inner[0] - inner[-1]) > 1.0:
                continue
            h = sg.Polygon(inner[:-1])
            if base.contains(h):
                holes.append(list(h.exterior.coords)[:-1])
        try:
            p = sg.Polygon(list(base.exterior.coords)[:-1], holes)
            p = shapely.make_valid(p.buffer(0))
        except Exception:  # noqa: BLE001
            continue
        if not p.is_empty and p.geom_type == "Polygon" and p.area >= 1.0:
            polys.append(p)
    return polys


def _linear_waterway(way: OSMWay, frame: Frame) -> sg.Polygon | None:
    lon = np.asarray([p[0] for p in way.geometry], dtype=np.float64)
    lat = np.asarray([p[1] for p in way.geometry], dtype=np.float64)
    east, north = frame.to_enu(lon, lat)
    coords = list(zip(east, north))
    if len(coords) < 2:
        return None
    line = sg.LineString(coords)
    if line.length < 1.0:
        return None
    width = parse_height(way.tags.get("width")) or 3.0
    poly = line.buffer(max(width, 1.0) / 2.0, cap_style="round", join_style="round")
    return poly if not poly.is_empty and poly.geom_type == "Polygon" else None


def _coastline_to_enu(way: OSMWay, frame: Frame) -> sg.LineString | None:
    lon = np.asarray([p[0] for p in way.geometry], dtype=np.float64)
    lat = np.asarray([p[1] for p in way.geometry], dtype=np.float64)
    east, north = frame.to_enu(lon, lat)
    coords = list(zip(east, north))
    if len(coords) < 2:
        return None
    line = sg.LineString(coords)
    return line if line.length > 0.5 else None


def _centroid_on_sea_side(poly: sg.Polygon, coast_lines: list[sg.LineString]) -> bool:
    """OSM convention: sea is on the right of a coastline in node order.

    Uses the nearest segment across all coastline lines to classify.
    """
    p = poly.representative_point()
    px, py = p.x, p.y
    best_d = float("inf")
    best_cross = 0.0
    for line in coast_lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            ax, ay = coords[i]
            bx, by = coords[i + 1]
            # Distance from point to segment
            dx, dy = bx - ax, by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= 1e-12:
                continue
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
            nx, ny = ax + t * dx, ay + t * dy
            d = np.hypot(px - nx, py - ny)
            if d < best_d:
                best_d = d
                # 2D cross: positive → left of edge direction, negative → right (sea)
                best_cross = dx * (py - ay) - dy * (px - ax)
    return best_cross < 0.0


def build_sea_polygon(osm: OSMData, frame: Frame,
                       radius_m: float) -> sg.Polygon | sg.MultiPolygon | None:
    coast_ways = osm.filter_ways(lambda t: t.get("natural") == "coastline")
    if not coast_ways:
        return None
    bbox = sg.box(-radius_m, -radius_m, radius_m, radius_m)
    coast_lines: list[sg.LineString] = []
    for w in coast_ways:
        line = _coastline_to_enu(w, frame)
        if line is None:
            continue
        clipped = line.intersection(bbox)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            coast_lines.append(clipped)
        elif clipped.geom_type == "MultiLineString":
            coast_lines.extend(clipped.geoms)
    if not coast_lines:
        return None

    # Polygonize bbox boundary + coastlines to enumerate all closed regions.
    merged = shapely.unary_union([bbox.boundary] + coast_lines)
    polygons = list(shapely.ops.polygonize(merged))
    sea_parts = [p for p in polygons if _centroid_on_sea_side(p, coast_lines)]
    if not sea_parts:
        return None
    if len(sea_parts) == 1:
        return sea_parts[0]
    return shapely.unary_union(sea_parts)


def _sea_mesh(sea_geom, sea_y: float, radius_m: float, max_step: float = 50.0):
    """Triangulate a sea polygon as a flat horizontal mesh at sea_y."""
    polys: list[sg.Polygon] = []
    if sea_geom.geom_type == "Polygon":
        polys = [sea_geom]
    elif sea_geom.geom_type == "MultiPolygon":
        polys = list(sea_geom.geoms)
    else:
        return None

    all_v = []
    all_i = []
    offset = 0
    for poly in polys:
        if poly.is_empty or poly.area < 1.0:
            continue
        ext = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
        inners = [np.asarray(h.coords, dtype=np.float64)[:-1] for h in poly.interiors]
        # Densify rings so the surface tessellation is uniform (nicer shading).
        rings = [_densify(ext, max_step)] + [_densify(r, max_step) for r in inners]
        flat = np.vstack(rings)
        ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        try:
            tri = triangulate_float64(flat, ring_ends).reshape(-1, 3)
        except Exception as exc:  # noqa: BLE001
            log.debug("sea earcut failed: %s", exc)
            continue
        if len(tri) == 0:
            continue
        v = np.stack([flat[:, 0].astype(np.float32),
                      np.full(len(flat), sea_y, dtype=np.float32),
                      (-flat[:, 1]).astype(np.float32)], axis=-1)
        all_v.append(v)
        all_i.append(tri.astype(np.uint32) + offset)
        offset += len(v)

    if not all_v:
        return None
    verts = np.concatenate(all_v, axis=0)
    idx = np.concatenate([i.ravel() for i in all_i], axis=0)
    norms = np.tile([0.0, 1.0, 0.0], (len(verts), 1)).astype(np.float32)
    return verts, norms, idx


def _densify(ring: np.ndarray, max_step: float) -> np.ndarray:
    out = [ring[0]]
    n = len(ring)
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        d = float(np.linalg.norm(b - a))
        if d > max_step:
            steps = int(np.ceil(d / max_step))
            for k in range(1, steps):
                out.append(a + (b - a) * (k / steps))
        if i < n - 1:
            out.append(b)
    return np.asarray(out, dtype=np.float64)


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler,
          *, radius_m: float, sea_y: float | None = None,
          sea_polygon: sg.Polygon | sg.MultiPolygon | None = None) -> WaterMesh:
    all_v: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    idx_offset = 0
    count = 0

    def add(poly, per_vertex):
        nonlocal idx_offset, count
        res = drape(poly, sampler, per_vertex=per_vertex, offset=0.02)
        if res is None:
            return
        v, n, i = res
        all_v.append(v)
        all_n.append(n)
        all_i.append(i + idx_offset)
        idx_offset += len(v)
        count += 1

    for w in osm.filter_ways(lambda t: t.get("natural") == "water"):
        p = _polygon_from_way(w, frame)
        if p is not None:
            add(p, per_vertex=False)

    for r in osm.filter_relations(lambda t: t.get("natural") == "water"):
        for p in _polygons_from_relation(r, frame):
            add(p, per_vertex=False)

    for w in osm.filter_ways(lambda t: "waterway" in t and t.get("waterway") not in ("dam", "weir", "lock_gate")):
        p = _linear_waterway(w, frame)
        if p is not None:
            add(p, per_vertex=True)

    # Sea from coastlines: one big flat mesh at absolute sea level.
    sea_poly = sea_polygon if sea_polygon is not None else build_sea_polygon(osm, frame, radius_m)
    if sea_poly is not None and sea_y is not None:
        sea = _sea_mesh(sea_poly, sea_y, radius_m)
        if sea is not None:
            v, n, i = sea
            all_v.append(v)
            all_n.append(n)
            all_i.append(i + idx_offset)
            idx_offset += len(v)
            count += 1
            log.info("sea mesh: %d verts, y=%.2f", len(v), sea_y)

    if not all_v:
        empty = np.zeros((0, 3), dtype=np.float32)
        return WaterMesh(empty, empty, np.zeros((0,), dtype=np.uint32), 0)
    return WaterMesh(
        vertices=np.concatenate(all_v, axis=0),
        normals=np.concatenate(all_n, axis=0),
        indices=np.concatenate(all_i, axis=0),
        count=count,
    )
