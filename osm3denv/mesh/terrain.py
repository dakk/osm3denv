"""Build a terrain mesh in the local ENU frame from SRTM heights.

Panda3D coordinate convention: x=east, y=north, z=up (right-handed, Z-up).

Sea/land boundary strategy
--------------------------
Instead of geometrically clipping triangles at the coastline (which requires
constrained Delaunay triangulation and introduces its own artefacts), we use
height clamping:

* Vertices that fall inside the sea polygon are pushed to a fixed depth below
  the sea plane.  The sea plane is opaque and fully occludes them.
* All terrain triangles are kept — no triangle is dropped or split.
* A DepthOffsetAttrib on the sea node (set in the renderer) prevents the last
  bit of Z-fighting right at the coastline boundary.

The clamped vertices create a natural underwater slope from the coast inward;
above the sea surface you see unmodified SRTM terrain up to the shoreline.
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
    heightmap: np.ndarray  # (G, G) float32 — zero-centred at origin
    radius_m: float
    origin_alt_m: float    # absolute SRTM altitude of the origin


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

    if sea_polygon is not None and not sea_polygon.is_empty:
        import shapely
        inside_sea = shapely.contains_xy(
            sea_polygon,
            east_g.ravel(), north_g.ravel(),
        ).reshape(grid, grid)

        # Sea plane sits at Panda3D z = -h0 - 0.3  (0.3 m below absolute sea
        # level, set in cli.py).  Clamp sea vertices to -h0 - 0.8 so they are
        # always 0.5 m below the sea plane regardless of SRTM noise.
        sea_clamp = float(-h0 - 0.8)
        heightmap = heightmap.copy()
        heightmap[inside_sea] = np.minimum(heightmap[inside_sea], sea_clamp)

        log.info(
            "sea clamp: pushed %d / %d vertices to %.1f m (below sea plane)",
            int(inside_sea.sum()), inside_sea.size, sea_clamp,
        )

    # Panda3D Z-up: (x=east, y=north, z=height).
    x = east_g.astype(np.float32).ravel()
    y = north_g.astype(np.float32).ravel()
    z = heightmap.ravel()
    vertices = np.stack([x, y, z], axis=-1)

    # Normals via numpy.gradient over the (possibly clamped) heightmap.
    # axis 0 = row (north decreases), axis 1 = col (east increases).
    gy_row, gx_col = np.gradient(heightmap, step_m, step_m)
    dh_dn = -gy_row   # dh/dnorth = -d/drow
    dh_de = gx_col
    norms = np.stack([-dh_de, -dh_dn, np.ones_like(dh_de)],
                     axis=-1).astype(np.float32).reshape(-1, 3)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True).clip(min=1e-8)

    # UVs: world-space planar, 1 unit = 1 metre.
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
    indices = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0).ravel()

    return TerrainData(vertices=vertices, normals=norms, indices=indices,
                       uvs=uvs, heightmap=heightmap, radius_m=radius_m,
                       origin_alt_m=h0)
