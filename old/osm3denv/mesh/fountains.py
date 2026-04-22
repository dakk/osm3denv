"""Fountains from OSM ``amenity=fountain`` ways/relations/nodes.

A fountain has two bits of geometry:
  * a low stone basin (the outer wall + top rim) rendered with the stone trim
    material;
  * a water disc filling the basin interior, rendered with the water material.

Polygon fountains use the OSM outline; point fountains get a parametric round
basin.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely.geometry as sg
from mapbox_earcut import triangulate_float64

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.geom import (parse_number, polygon_from_way,
                                polygons_from_relation)
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

BASIN_HEIGHT = 0.45        # outer wall height in metres
WALL_THICKNESS = 0.30      # basin wall thickness
WATER_DROP = 0.06          # water surface this far below rim
POINT_RADIUS = 1.6         # assumed radius for a point-fountain basin
POINT_SEGMENTS = 24


@dataclass
class FountainsMesh:
    # Stone basin (walls + top rim)
    stone_vertices: np.ndarray
    stone_normals: np.ndarray
    stone_indices: np.ndarray
    stone_uvs: np.ndarray
    # Water disc at top of basin
    water_vertices: np.ndarray
    water_normals: np.ndarray
    water_indices: np.ndarray
    water_uvs: np.ndarray
    count: int


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _add_wall_strip(ring_xy: np.ndarray, base_y: float, top_y: float,
                    verts, norms, uvs, idx, v_off: int,
                    *, outward: bool) -> int:
    """Extrude a closed ring into a vertical wall strip.

    ``ring_xy`` is a (N, 2) array of (east, north) and is assumed closed
    (first != last; we wrap). ``outward=True`` → normals point away from the
    ring centroid; ``outward=False`` → inward (for the inside of a basin).
    """
    n = len(ring_xy)
    if n < 3:
        return v_off
    centroid = ring_xy.mean(axis=0)
    # Cumulative along-ring distance for V coordinate.
    cum = 0.0
    for i in range(n):
        a = ring_xy[i]
        b = ring_xy[(i + 1) % n]
        seg = b - a
        seg_len = float(np.hypot(seg[0], seg[1]))
        if seg_len < 1e-4:
            continue
        # Outward perpendicular (rotate segment 90° to the right of direction a→b).
        # shapely polygons coming out of _take_polygon are CCW-oriented, so
        # rotating the forward tangent 90° clockwise points outward.
        tx, ty = seg / seg_len
        nx, ny = ty, -tx
        # Reference it against the centroid to correct sign if the polygon is
        # non-convex at this edge (the earcut output handled CCW ordering but
        # a concave section can flip the naive perpendicular).
        mid = (a + b) * 0.5
        to_mid = mid - centroid
        if (nx * to_mid[0] + ny * to_mid[1]) < 0:
            nx, ny = -nx, -ny
        if not outward:
            nx, ny = -nx, -ny
        # Four corners (Ogre coords: east, y, -north).
        a_bot = (float(a[0]),  base_y, -float(a[1]))
        b_bot = (float(b[0]),  base_y, -float(b[1]))
        a_top = (float(a[0]),  top_y,  -float(a[1]))
        b_top = (float(b[0]),  top_y,  -float(b[1]))
        normal = (float(nx), 0.0, -float(ny))
        v_a, v_b = cum, cum + seg_len
        h = top_y - base_y
        # Two triangles per segment; CCW when viewed from outside.
        if outward:
            verts.extend([a_bot, b_bot, b_top, a_top])
        else:
            verts.extend([a_bot, a_top, b_top, b_bot])
        norms.extend([normal] * 4)
        uvs.extend([(v_a, 0.0), (v_b, 0.0), (v_b, h), (v_a, h)])
        idx.extend([(v_off, v_off + 1, v_off + 2),
                    (v_off, v_off + 2, v_off + 3)])
        v_off += 4
        cum += seg_len
    return v_off


def _triangulate_ring_annulus(outer: np.ndarray, inner: np.ndarray,
                               y: float, verts, norms, uvs, idx,
                               v_off: int) -> int:
    """Triangulate the top rim (ring between ``outer`` and ``inner`` polygons).

    ``outer`` and ``inner`` are (N, 2) arrays in ENU. Earcut handles the hole.
    """
    flat = np.vstack([outer, inner])
    ring_ends = np.asarray([len(outer), len(outer) + len(inner)],
                           dtype=np.uint32)
    try:
        tri = triangulate_float64(flat, ring_ends).reshape(-1, 3)
    except Exception as exc:  # noqa: BLE001
        log.debug("basin rim earcut failed: %s", exc)
        return v_off
    if len(tri) == 0:
        return v_off
    base = v_off
    for p in flat:
        verts.append((float(p[0]), y, -float(p[1])))
        norms.append((0.0, 1.0, 0.0))
        uvs.append((float(p[0]) * 0.5, -float(p[1]) * 0.5))
    for a, b, c in tri:
        idx.append((base + int(a), base + int(b), base + int(c)))
    return v_off + len(flat)


def _triangulate_polygon_fill(ring: np.ndarray, y: float,
                               verts, norms, uvs, idx, v_off: int) -> int:
    """Triangulate a single polygon ring as a flat fill at height ``y``."""
    ring_ends = np.asarray([len(ring)], dtype=np.uint32)
    try:
        tri = triangulate_float64(ring, ring_ends).reshape(-1, 3)
    except Exception as exc:  # noqa: BLE001
        log.debug("basin fill earcut failed: %s", exc)
        return v_off
    if len(tri) == 0:
        return v_off
    base = v_off
    for p in ring:
        verts.append((float(p[0]), y, -float(p[1])))
        norms.append((0.0, 1.0, 0.0))
        uvs.append((float(p[0]) * 0.5, -float(p[1]) * 0.5))
    for a, b, c in tri:
        idx.append((base + int(a), base + int(b), base + int(c)))
    return v_off + len(ring)


def _build_polygon_fountain(poly: sg.Polygon, sampler: TerrainSampler,
                             stone_v, stone_n, stone_u, stone_i, s_off,
                             water_v, water_n, water_u, water_i, w_off):
    """Build basin geometry from a shapely polygon."""
    inner = poly.buffer(-WALL_THICKNESS)
    if inner.is_empty or inner.geom_type != "Polygon" or inner.area < 0.5:
        return s_off, w_off, False

    cx, cy = poly.centroid.x, poly.centroid.y
    base_y = float(sampler.height_at(cx, cy))
    top_y = base_y + BASIN_HEIGHT
    water_y = top_y - WATER_DROP

    outer_ring = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
    inner_ring = np.asarray(inner.exterior.coords, dtype=np.float64)[:-1]

    # Outer wall (normal outward), inner wall (normal inward), top rim.
    s_off = _add_wall_strip(outer_ring, base_y, top_y,
                            stone_v, stone_n, stone_u, stone_i, s_off,
                            outward=True)
    s_off = _add_wall_strip(inner_ring, base_y, top_y,
                            stone_v, stone_n, stone_u, stone_i, s_off,
                            outward=False)
    s_off = _triangulate_ring_annulus(outer_ring, inner_ring, top_y,
                                      stone_v, stone_n, stone_u, stone_i,
                                      s_off)

    # Water fills the inner polygon just below the rim.
    w_off = _triangulate_polygon_fill(inner_ring, water_y,
                                      water_v, water_n, water_u, water_i,
                                      w_off)
    return s_off, w_off, True


def _build_point_fountain(east: float, north: float, sampler: TerrainSampler,
                           stone_v, stone_n, stone_u, stone_i, s_off,
                           water_v, water_n, water_u, water_i, w_off):
    """Build a round basin at a single lon/lat (no polygon)."""
    base_y = float(sampler.height_at(east, north))
    top_y = base_y + BASIN_HEIGHT
    water_y = top_y - WATER_DROP

    angles = np.linspace(0.0, 2.0 * np.pi, POINT_SEGMENTS, endpoint=False)
    outer_ring = np.stack([east + POINT_RADIUS * np.cos(angles),
                           north + POINT_RADIUS * np.sin(angles)], axis=-1)
    inner_ring = np.stack([east + (POINT_RADIUS - WALL_THICKNESS) * np.cos(angles),
                           north + (POINT_RADIUS - WALL_THICKNESS) * np.sin(angles)],
                          axis=-1)

    s_off = _add_wall_strip(outer_ring, base_y, top_y,
                            stone_v, stone_n, stone_u, stone_i, s_off,
                            outward=True)
    s_off = _add_wall_strip(inner_ring, base_y, top_y,
                            stone_v, stone_n, stone_u, stone_i, s_off,
                            outward=False)
    s_off = _triangulate_ring_annulus(outer_ring, inner_ring, top_y,
                                      stone_v, stone_n, stone_u, stone_i,
                                      s_off)
    w_off = _triangulate_polygon_fill(inner_ring, water_y,
                                      water_v, water_n, water_u, water_i,
                                      w_off)
    return s_off, w_off, True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _empty() -> FountainsMesh:
    z3 = np.zeros((0, 3), dtype=np.float32)
    z2 = np.zeros((0, 2), dtype=np.float32)
    zi = np.zeros((0,), dtype=np.uint32)
    return FountainsMesh(z3, z3, zi, z2, z3, z3, zi, z2, 0)


def _finalise(stone_v, stone_n, stone_u, stone_i,
              water_v, water_n, water_u, water_i, count: int) -> FountainsMesh:
    if count == 0:
        return _empty()
    return FountainsMesh(
        stone_vertices=np.asarray(stone_v, dtype=np.float32),
        stone_normals=np.asarray(stone_n, dtype=np.float32),
        stone_indices=np.asarray(stone_i, dtype=np.uint32).reshape(-1),
        stone_uvs=np.asarray(stone_u, dtype=np.float32),
        water_vertices=np.asarray(water_v, dtype=np.float32),
        water_normals=np.asarray(water_n, dtype=np.float32),
        water_indices=np.asarray(water_i, dtype=np.uint32).reshape(-1),
        water_uvs=np.asarray(water_u, dtype=np.float32),
        count=count,
    )


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler) -> FountainsMesh:
    stone_v, stone_n, stone_u, stone_i = [], [], [], []
    water_v, water_n, water_u, water_i = [], [], [], []
    s_off = w_off = 0
    count = 0
    used_polygon_keys: set[int] = set()   # OSM ids we've already rendered

    # Polygon fountains from ways.
    for w in osm.filter_ways(lambda t: t.get("amenity") == "fountain"):
        poly = polygon_from_way(w, frame, min_area=0.5)
        if poly is None:
            continue
        s_off, w_off, ok = _build_polygon_fountain(
            poly, sampler,
            stone_v, stone_n, stone_u, stone_i, s_off,
            water_v, water_n, water_u, water_i, w_off)
        if ok:
            count += 1
            used_polygon_keys.add(w.id)

    # Polygon fountains from multipolygon relations.
    for r in osm.filter_relations(lambda t: t.get("amenity") == "fountain"):
        for poly in polygons_from_relation(r, frame, min_area=0.5):
            s_off, w_off, ok = _build_polygon_fountain(
                poly, sampler,
                stone_v, stone_n, stone_u, stone_i, s_off,
                water_v, water_n, water_u, water_i, w_off)
            if ok:
                count += 1

    # Point fountains — only if no polygon exists very nearby (avoid double
    # basins for cases where OSM has both a node and a way for the same feature).
    point_nodes = osm.filter_nodes(lambda t: t.get("amenity") == "fountain")
    for node in point_nodes:
        e, n = frame.to_enu(node.lon, node.lat)
        e = float(e); n = float(n)
        # Parametric radius override if the OSM node declares one.
        s_off, w_off, ok = _build_point_fountain(
            e, n, sampler,
            stone_v, stone_n, stone_u, stone_i, s_off,
            water_v, water_n, water_u, water_i, w_off)
        if ok:
            count += 1

    if count:
        log.info("fountains: %d basins", count)

    return _finalise(stone_v, stone_n, stone_u, stone_i,
                     water_v, water_n, water_u, water_i, count)
