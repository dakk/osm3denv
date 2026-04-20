"""Shared helper: drape a 2D polygon onto the terrain as a flat ribbon mesh.

Triangulates with earcut; returns Ogre-frame vertices/normals/indices.
"""
from __future__ import annotations

import logging

import numpy as np
import shapely
import shapely.geometry as sg
from mapbox_earcut import triangulate_float64

from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)


def planar_uv(vertices: np.ndarray, tile_m: float = 4.0) -> np.ndarray:
    """World-space planar UVs: U = east / tile, V = north / tile.

    In our Ogre mapping vertex[:, 2] = -north so we negate to recover north.
    """
    u = vertices[:, 0] / tile_m
    v = -vertices[:, 2] / tile_m
    return np.stack([u, v], axis=-1).astype(np.float32)


def _densify_ring(ring: np.ndarray, max_step: float) -> np.ndarray:
    out = [ring[0]]
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        d = np.linalg.norm(b - a)
        if d > max_step:
            n = int(np.ceil(d / max_step))
            for k in range(1, n):
                t = k / n
                out.append(a + (b - a) * t)
        out.append(b)
    return np.asarray(out, dtype=np.float64)


def drape(poly: sg.Polygon, sampler: TerrainSampler,
          *, per_vertex: bool = True, flat_y: float | None = None,
          offset: float = 0.05, max_step: float = 2.0):
    """Triangulate ``poly`` (ENU coords) and return an Ogre-frame mesh.

    If ``per_vertex`` is True, each vertex y is sampled from the terrain
    (used for roads). Otherwise all vertices share ``flat_y`` (used for water
    areas whose surface is assumed locally flat).
    """
    if poly.is_empty or poly.area < 0.1:
        return None

    exterior = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
    interiors = [np.asarray(r.coords, dtype=np.float64)[:-1] for r in poly.interiors]

    rings = [_densify_ring(_close_if_needed(exterior), max_step)]
    for r in interiors:
        rings.append(_densify_ring(_close_if_needed(r), max_step))

    flat2d = np.vstack(rings)
    ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        tri = triangulate_float64(flat2d, ring_ends).reshape(-1, 3)
    except Exception as exc:  # noqa: BLE001
        log.debug("drape earcut failed: %s", exc)
        return None
    if len(tri) == 0:
        return None

    if per_vertex:
        y = sampler.height_at(flat2d[:, 0], flat2d[:, 1]) + offset
    else:
        if flat_y is None:
            cx, cy = poly.centroid.x, poly.centroid.y
            flat_y = float(sampler.height_at(cx, cy))
        y = np.full(len(flat2d), flat_y + offset, dtype=np.float32)

    vertices = np.stack([flat2d[:, 0].astype(np.float32),
                         y.astype(np.float32),
                         (-flat2d[:, 1]).astype(np.float32)], axis=-1)
    normals = np.tile([0.0, 1.0, 0.0], (len(vertices), 1)).astype(np.float32)
    indices = tri.astype(np.uint32).ravel()
    return vertices, normals, indices


def _close_if_needed(ring: np.ndarray) -> np.ndarray:
    if len(ring) >= 2 and not np.allclose(ring[0], ring[-1]):
        return ring
    return ring[:-1] if len(ring) > 1 and np.allclose(ring[0], ring[-1]) else ring
