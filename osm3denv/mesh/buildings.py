"""Extruded building meshes from OSM footprints."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely.geometry as sg
from mapbox_earcut import triangulate_float64

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.geom import (
    parse_number as parse_height,
    polygon_from_relation,
    polygon_from_way,
)
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)


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
    uvs: np.ndarray
    count: int




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
    wall_uvs: list[tuple[float, float]] = []
    wall_idx: list[tuple[int, int, int]] = []
    base_offset = 2 * n
    # Brick texture: 1 unit = 1 meter for both axes.
    WALL_TILE = 1.0
    for ring in rings:
        # Normalize to CCW for wall generation: outward normal = right of edge,
        # triangle winding (a_b, b_b, b_t, a_t) is CCW when viewed from outside.
        signed_area = 0.0
        m = len(ring)
        for i in range(m):
            signed_area += ring[i][0] * ring[(i + 1) % m][1] \
                         - ring[(i + 1) % m][0] * ring[i][1]
        if signed_area < 0:
            ring = ring[::-1]
        m = len(ring)
        u_running = 0.0  # cumulative horizontal length along the ring
        top_v = height_m / WALL_TILE
        for i in range(m):
            a = ring[i]
            b = ring[(i + 1) % m]
            de = b[0] - a[0]
            dn = b[1] - a[1]
            length = float(np.hypot(de, dn))
            if length < 1e-6:
                continue
            normal = (dn / length, 0.0, de / length)
            a_b = (float(a[0]), base_y, float(-a[1]))
            b_b = (float(b[0]), base_y, float(-b[1]))
            b_t = (float(b[0]), base_y + height_m, float(-b[1]))
            a_t = (float(a[0]), base_y + height_m, float(-a[1]))
            u0 = u_running / WALL_TILE
            u1 = (u_running + length) / WALL_TILE
            v0 = base_offset + len(wall_verts)
            wall_verts.extend([a_b, b_b, b_t, a_t])
            wall_norms.extend([normal] * 4)
            wall_uvs.extend([(u0, 0.0), (u1, 0.0), (u1, top_v), (u0, top_v)])
            wall_idx.extend([(v0, v0 + 1, v0 + 2), (v0, v0 + 2, v0 + 3)])
            u_running += length

    if wall_verts:
        wv = np.asarray(wall_verts, dtype=np.float32)
        wn = np.asarray(wall_norms, dtype=np.float32)
        wu = np.asarray(wall_uvs, dtype=np.float32)
        wi = np.asarray(wall_idx, dtype=np.uint32).reshape(-1)
    else:
        wv = np.zeros((0, 3), dtype=np.float32)
        wn = np.zeros((0, 3), dtype=np.float32)
        wu = np.zeros((0, 2), dtype=np.float32)
        wi = np.zeros((0,), dtype=np.uint32)

    # Roof/floor UVs: planar world-space, 1 meter per unit (material scales).
    ROOF_TILE = 1.0
    roof_uvs = np.stack([flat2d[:, 0] / ROOF_TILE,
                         flat2d[:, 1] / ROOF_TILE], axis=-1).astype(np.float32)
    floor_uvs = roof_uvs.copy()

    vertices = np.concatenate([roof, floor, wv], axis=0)
    normals = np.concatenate([roof_norms, floor_norms, wn], axis=0)
    uvs = np.concatenate([roof_uvs, floor_uvs, wu], axis=0)
    indices = np.concatenate([roof_idx.ravel(), floor_idx.ravel(), wi], axis=0)
    return vertices, normals, uvs, indices


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler) -> BuildingsMesh:
    all_v: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    index_offset = 0
    count = 0

    def add(tuple_):
        nonlocal index_offset, count
        v, n, u, i = tuple_
        all_v.append(v)
        all_n.append(n)
        all_u.append(u)
        all_i.append(i + index_offset)
        index_offset += len(v)
        count += 1

    for w in osm.filter_ways(lambda t: "building" in t or "building:part" in t):
        poly = polygon_from_way(w, frame)
        if poly is None:
            continue
        h = resolve_height(w.tags)
        cx, cy = poly.centroid.x, poly.centroid.y
        base_y = float(sampler.height_at(cx, cy))
        ext = _extrude(poly, h, base_y)
        if ext is not None:
            add(ext)

    for r in osm.filter_relations(lambda t: "building" in t):
        poly = polygon_from_relation(r, frame)
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
        empty_uv = np.zeros((0, 2), dtype=np.float32)
        return BuildingsMesh(empty, empty, np.zeros((0,), dtype=np.uint32),
                             empty_uv, 0)

    return BuildingsMesh(
        vertices=np.concatenate(all_v, axis=0),
        normals=np.concatenate(all_n, axis=0),
        indices=np.concatenate(all_i, axis=0),
        uvs=np.concatenate(all_u, axis=0),
        count=count,
    )
