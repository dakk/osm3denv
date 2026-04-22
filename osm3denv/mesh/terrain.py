"""Build a terrain mesh in the local ENU frame from SRTM heights.

Panda3D coordinate convention: x=east, y=north, z=up (right-handed, Z-up).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from osm3denv.frame import Frame


@dataclass
class TerrainData:
    vertices: np.ndarray   # (N, 3) float32  — (east, north, height)
    normals: np.ndarray    # (N, 3) float32
    indices: np.ndarray    # (T*3,) uint32
    uvs: np.ndarray        # (N, 2) float32
    heightmap: np.ndarray  # (G, G) float32 — zero-centered at origin
    radius_m: float
    origin_alt_m: float    # absolute SRTM altitude of the origin


def build(*, frame: Frame, radius_m: float, grid: int,
          hgt_loader: Callable) -> TerrainData:
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
    # heightmap axis 0 = row (north decreases), axis 1 = col (east increases).
    # gradient returns (d/drow, d/dcol) scaled by spacing.
    gy_row, gx_col = np.gradient(heightmap, step_m, step_m)
    # dh/dnorth = -gy_row (north decreases along rows).
    # dh/deast  =  gx_col.
    dh_de = gx_col
    dh_dn = -gy_row
    nx = -dh_de
    ny = -dh_dn
    nz = np.ones_like(dh_de)
    norms = np.stack([nx, ny, nz], axis=-1).astype(np.float32).reshape(-1, 3)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True).clip(min=1e-8)

    # Triangle indices (two tris per quad, CCW when viewed from +z).
    g = grid
    r = np.arange(g - 1, dtype=np.uint32)[:, None]
    c = np.arange(g - 1, dtype=np.uint32)[None, :]
    i00 = r * g + c           # top-left (north+, east-)
    i10 = (r + 1) * g + c     # bottom-left (north-, east-)
    i11 = (r + 1) * g + (c + 1)  # bottom-right
    i01 = r * g + (c + 1)     # top-right
    # CCW from +z looking down: (i10, i11, i01) and (i10, i01, i00).
    tri1 = np.stack([i10, i11, i01], axis=-1)
    tri2 = np.stack([i10, i01, i00], axis=-1)
    indices = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0).ravel()

    # UVs: world-space planar, 1 unit = 1 meter.
    uvs = np.stack([vertices[:, 0], vertices[:, 1]], axis=-1).astype(np.float32)

    return TerrainData(vertices=vertices, normals=norms, indices=indices,
                       uvs=uvs, heightmap=heightmap, radius_m=radius_m,
                       origin_alt_m=h0)
