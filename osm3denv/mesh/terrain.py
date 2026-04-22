"""Build a terrain mesh in the local ENU frame from SRTM heights.

Panda3D coordinate convention: x=east, y=north, z=up (right-handed, Z-up).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from osm3denv.frame import Frame

log = logging.getLogger(__name__)


@dataclass
class TerrainData:
    vertices: np.ndarray   # (N, 3) float32  — (east, north, height)
    normals: np.ndarray    # (N, 3) float32
    indices: np.ndarray    # (T*3,) uint32
    uvs: np.ndarray        # (N, 2) float32
    heightmap: np.ndarray  # (G, G) float32 — zero-centered at origin
    radius_m: float
    origin_alt_m: float    # absolute SRTM altitude of the origin


def _grid_coords(x: float, y: float, grid: int, radius_m: float) -> tuple[float, float]:
    """Convert ENU (x=east, y=north) to fractional grid (row, col)."""
    col = (x + radius_m) / (2.0 * radius_m) * (grid - 1)
    row = (radius_m - y) / (2.0 * radius_m) * (grid - 1)
    return float(np.clip(row, 0, grid - 1)), float(np.clip(col, 0, grid - 1))


def _bilinear(arr: np.ndarray, row_f: float, col_f: float, grid: int) -> float:
    r0 = int(row_f)
    c0 = int(col_f)
    r1 = min(r0 + 1, grid - 1)
    c1 = min(c0 + 1, grid - 1)
    fr, fc = row_f - r0, col_f - c0
    return float(
        arr[r0, c0] * (1 - fr) * (1 - fc) +
        arr[r0, c1] * (1 - fr) * fc +
        arr[r1, c0] * fr * (1 - fc) +
        arr[r1, c1] * fr * fc
    )


def _interp_height(x: float, y: float, heightmap: np.ndarray,
                   grid: int, radius_m: float) -> float:
    row, col = _grid_coords(x, y, grid, radius_m)
    return _bilinear(heightmap, row, col, grid)


def _interp_normal(x: float, y: float, dh_de: np.ndarray, dh_dn: np.ndarray,
                   grid: int, radius_m: float) -> tuple[float, float, float]:
    row, col = _grid_coords(x, y, grid, radius_m)
    de = _bilinear(dh_de, row, col, grid)
    dn = _bilinear(dh_dn, row, col, grid)
    nx, ny, nz = -de, -dn, 1.0
    L = float(np.sqrt(nx * nx + ny * ny + nz * nz))
    if L < 1e-8:
        L = 1.0
    return nx / L, ny / L, nz / L


def _triangulate_poly(poly) -> list[list[tuple[float, float]]]:
    """Triangulate a shapely Polygon; return [(x,y),(x,y),(x,y)] triples in CCW order."""
    import shapely
    tris_geom = shapely.delaunay_triangles(poly)
    result = []
    for tri in tris_geom.geoms:
        if not poly.contains(tri.centroid):
            continue
        coords = list(tri.exterior.coords)[:-1]
        if len(coords) != 3:
            continue
        # Enforce CCW winding viewed from +z
        (x0, y0), (x1, y1), (x2, y2) = coords
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if cross < 0:
            coords = [coords[0], coords[2], coords[1]]
        result.append(coords)
    return result


def _clip_to_land(
    verts: np.ndarray,      # (N, 3) original grid vertices
    norms: np.ndarray,      # (N, 3) original grid normals
    uvs: np.ndarray,        # (N, 2) original grid UVs
    tris: np.ndarray,       # (T, 3) uint32 triangle index array
    land_poly,              # shapely Polygon/MultiPolygon — the land area
    inside_sea: np.ndarray, # (N,) bool — True if vertex is inside sea
    heightmap: np.ndarray,
    dh_de: np.ndarray,
    dh_dn: np.ndarray,
    grid: int,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip the terrain to the land polygon.

    Triangles fully on land are kept as-is (fast path, vectorised).
    Triangles fully in the sea are dropped.
    Boundary triangles are intersected with land_poly; the result is
    triangulated and new vertices are interpolated from the heightmap.
    """
    import shapely.geometry as sg

    # Vectorised classification of each triangle.
    tri_sea = inside_sea[tris]        # (T, 3) bool
    all_land_mask = ~tri_sea.any(axis=1)  # (T,) — all 3 vertices on land
    all_sea_mask = tri_sea.all(axis=1)    # (T,) — all 3 vertices in sea
    mixed_mask = ~(all_land_mask | all_sea_mask)

    land_tris = tris[all_land_mask]   # kept unchanged

    log.info(
        "coast clip: %d land / %d sea / %d boundary triangles",
        int(all_land_mask.sum()), int(all_sea_mask.sum()), int(mixed_mask.sum())
    )

    # Accumulators for new vertices introduced at the coastline.
    new_v: list[list[float]] = []
    new_n: list[list[float]] = []
    new_u: list[list[float]] = []
    new_t: list[list[int]] = []
    vi = len(verts)  # index of the first new vertex

    for tri in tris[mixed_mask]:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        poly = sg.Polygon([verts[a, :2], verts[b, :2], verts[c, :2]])
        try:
            clipped = poly.intersection(land_poly)
        except Exception:
            continue
        if clipped.is_empty:
            continue

        if clipped.geom_type == "Polygon":
            parts = [clipped]
        elif clipped.geom_type == "MultiPolygon":
            parts = list(clipped.geoms)
        elif clipped.geom_type == "GeometryCollection":
            parts = [g for g in clipped.geoms if g.geom_type == "Polygon"]
        else:
            continue

        for part in parts:
            if part.area < 1.0:      # skip slivers < 1 m²
                continue
            for coords in _triangulate_poly(part):
                idxs = []
                for (x, y) in coords:
                    z = _interp_height(x, y, heightmap, grid, radius_m)
                    nx, ny, nz = _interp_normal(x, y, dh_de, dh_dn, grid, radius_m)
                    new_v.append([float(x), float(y), float(z)])
                    new_n.append([float(nx), float(ny), float(nz)])
                    new_u.append([float(x), float(y)])
                    idxs.append(vi)
                    vi += 1
                new_t.append(idxs)

    if new_v:
        extra_v = np.array(new_v, dtype=np.float32)
        extra_n = np.array(new_n, dtype=np.float32)
        extra_n /= np.linalg.norm(extra_n, axis=1, keepdims=True).clip(min=1e-8)
        extra_u = np.array(new_u, dtype=np.float32)
        extra_t = np.array(new_t, dtype=np.uint32)

        all_verts = np.concatenate([verts, extra_v], axis=0)
        all_norms = np.concatenate([norms, extra_n], axis=0)
        all_uvs = np.concatenate([uvs, extra_u], axis=0)
        all_tris = np.concatenate([land_tris, extra_t], axis=0)
    else:
        all_verts, all_norms, all_uvs, all_tris = verts, norms, uvs, land_tris

    return all_verts, all_norms, all_uvs, all_tris


def build(*, frame: Frame, radius_m: float, grid: int,
          hgt_loader: Callable, sea_polygon=None) -> TerrainData:
    step_m = 2.0 * radius_m / (grid - 1)
    bbox = frame.bbox_ll(radius_m, pad_m=step_m)
    mosaic = hgt_loader(bbox)

    # ENU grid: row 0 is north_max, col 0 is east_min.
    eastings = np.linspace(-radius_m, radius_m, grid)
    northings = np.linspace(radius_m, -radius_m, grid)
    east_g, north_g = np.meshgrid(eastings, northings)  # (G, G)
    lon_g, lat_g = frame.to_ll(east_g, north_g)
    heights = mosaic.sample(lat_g, lon_g).astype(np.float32)

    # Zero at origin so the spawn point has z≈0.
    h0 = float(mosaic.sample(np.array([frame.lat0]), np.array([frame.lon0]))[0])
    heightmap = heights - h0

    # Panda3D Z-up: (x=east, y=north, z=height).
    x = east_g.astype(np.float32).ravel()
    y = north_g.astype(np.float32).ravel()
    z = heightmap.ravel()
    vertices = np.stack([x, y, z], axis=-1)

    # Normals via numpy.gradient over the heightmap.
    gy_row, gx_col = np.gradient(heightmap, step_m, step_m)
    dh_de = gx_col
    dh_dn = -gy_row
    nx_g = -dh_de
    ny_g = -dh_dn
    nz_g = np.ones_like(dh_de)
    norms = np.stack([nx_g, ny_g, nz_g], axis=-1).astype(np.float32).reshape(-1, 3)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True).clip(min=1e-8)

    # UVs: world-space planar, 1 unit = 1 meter.
    uvs = np.stack([vertices[:, 0], vertices[:, 1]], axis=-1).astype(np.float32)

    # Triangle indices (two tris per quad, CCW when viewed from +z).
    g = grid
    r = np.arange(g - 1, dtype=np.uint32)[:, None]
    c = np.arange(g - 1, dtype=np.uint32)[None, :]
    i00 = r * g + c
    i10 = (r + 1) * g + c
    i11 = (r + 1) * g + (c + 1)
    i01 = r * g + (c + 1)
    tri1 = np.stack([i10, i11, i01], axis=-1)
    tri2 = np.stack([i10, i01, i00], axis=-1)
    tris = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0)

    if sea_polygon is not None and not sea_polygon.is_empty:
        import shapely
        import shapely.geometry as sg

        inside_sea = shapely.contains_xy(
            sea_polygon,
            east_g.ravel(), north_g.ravel(),
        )
        log.info(
            "sea mask: %d / %d vertices inside sea polygon",
            int(inside_sea.sum()), inside_sea.size
        )

        bbox_poly = sg.box(-radius_m, -radius_m, radius_m, radius_m)
        land_poly = bbox_poly.difference(sea_polygon)

        vertices, norms, uvs, tris = _clip_to_land(
            vertices, norms, uvs, tris,
            land_poly, inside_sea,
            heightmap, dh_de, dh_dn, grid, radius_m,
        )
        log.info("terrain: %d verts, %d tris after coastline clip",
                 len(vertices), len(tris))

    indices = tris.ravel()

    return TerrainData(vertices=vertices, normals=norms, indices=indices,
                       uvs=uvs, heightmap=heightmap, radius_m=radius_m,
                       origin_alt_m=h0)
