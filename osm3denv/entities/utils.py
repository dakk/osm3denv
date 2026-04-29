"""Shared geometry utilities."""
from __future__ import annotations

import numpy as np


def grid_coords(x: float, y: float, grid: int, radius_m: float) -> tuple[float, float]:
    col = (x + radius_m) / (2.0 * radius_m) * (grid - 1)
    row = (radius_m - y) / (2.0 * radius_m) * (grid - 1)
    return float(np.clip(row, 0, grid - 1)), float(np.clip(col, 0, grid - 1))


def bilinear(arr: np.ndarray, row_f: float, col_f: float, grid: int) -> float:
    r0 = min(int(row_f), grid - 2)
    c0 = min(int(col_f), grid - 2)
    fr, fc = row_f - r0, col_f - c0
    return float(
        arr[r0,   c0  ] * (1 - fr) * (1 - fc) +
        arr[r0,   c0+1] * (1 - fr) * fc        +
        arr[r0+1, c0  ] * fr       * (1 - fc)  +
        arr[r0+1, c0+1] * fr       * fc
    )


def sample_z(x: float, y: float,
             heightmap: np.ndarray, grid: int, radius_m: float) -> float:
    row, col = grid_coords(x, y, grid, radius_m)
    return bilinear(heightmap, row, col, grid)


def sample_z_vec(x_arr, y_arr,
                 heightmap: np.ndarray, grid: int, radius_m: float) -> np.ndarray:
    """Vectorised bilinear terrain-height sampling; returns float32 array."""
    scale = (grid - 1) / (2.0 * radius_m)
    col_f = np.clip((np.asarray(x_arr, np.float64) + radius_m) * scale, 0.0, grid - 1)
    row_f = np.clip((radius_m - np.asarray(y_arr, np.float64)) * scale, 0.0, grid - 1)
    r0 = np.minimum(row_f.astype(np.int32), grid - 2)
    c0 = np.minimum(col_f.astype(np.int32), grid - 2)
    fr = row_f - r0
    fc = col_f - c0
    return (
        heightmap[r0,   c0  ] * (1 - fr) * (1 - fc) +
        heightmap[r0,   c0+1] * (1 - fr) * fc        +
        heightmap[r0+1, c0  ] * fr        * (1 - fc) +
        heightmap[r0+1, c0+1] * fr        * fc
    ).astype(np.float32)


def triangulate_flat_poly(poly, max_seg: float) -> list[list[tuple[float, float]]]:
    """Densify *poly*, Delaunay-triangulate, return CCW (x, y) triples."""
    import shapely
    densified = poly.segmentize(max_seg)
    result = []
    for tri in shapely.delaunay_triangles(densified).geoms:
        if not poly.covers(tri.centroid):
            continue
        coords = list(tri.exterior.coords)[:-1]
        if len(coords) != 3:
            continue
        (x0, y0), (x1, y1), (x2, y2) = coords
        if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0:
            coords = [coords[0], coords[2], coords[1]]
        result.append(coords)
    return result
