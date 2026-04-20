"""Build a terrain mesh in the local ENU frame from SRTM heights.

Ogre coordinate convention used here: x=east, y=height, z=−north.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import shapely

from osm3denv.frame import Frame
from osm3denv.mesh.sample import TerrainSampler


@dataclass
class TerrainData:
    vertices: np.ndarray   # (N, 3) float32
    normals: np.ndarray    # (N, 3) float32
    indices: np.ndarray    # (T*3,) uint32
    heightmap: np.ndarray  # (G, G) float32 — zero-centered at origin
    radius_m: float
    sampler: TerrainSampler
    origin_alt_m: float    # absolute SRTM altitude of the origin; sea y = -origin_alt_m


def build(*, frame: Frame, radius_m: float, grid: int,
          hgt_loader: Callable, sea_mask=None) -> TerrainData:
    # Pad bbox by one grid step so sampling doesn't clip at the edge.
    step_m = 2.0 * radius_m / (grid - 1)
    # Fetch SRTM for the geographic bbox enclosing the ENU square.
    bbox = frame.bbox_ll(radius_m, pad_m=step_m)
    mosaic = hgt_loader(bbox)

    # ENU grid: row 0 is north_max, col 0 is east_min.
    eastings = np.linspace(-radius_m, radius_m, grid)
    northings = np.linspace(radius_m, -radius_m, grid)
    east_g, north_g = np.meshgrid(eastings, northings)  # shape (G, G)
    lon_g, lat_g = frame.to_ll(east_g, north_g)
    heights = mosaic.sample(lat_g, lon_g).astype(np.float32)

    # Zero at origin so the spawn point has y≈0.
    h0 = float(mosaic.sample(np.array([frame.lat0]), np.array([frame.lon0]))[0])

    # Clamp sea cells below absolute sea level so the sea plane fully occludes
    # any SRTM noise / filled-void artifacts over water.
    if sea_mask is not None and not sea_mask.is_empty:
        inside = shapely.contains_xy(sea_mask,
                                     east_g.ravel(), north_g.ravel()).reshape(heights.shape)
        heights = np.where(inside, np.float32(-1.0), heights)  # ~1 m below abs 0

    heightmap = heights - h0

    # Vertices, Ogre mapping (x=east, y=height, z=-north).
    x = east_g.astype(np.float32).ravel()
    y = heightmap.ravel()
    z = (-north_g).astype(np.float32).ravel()
    vertices = np.stack([x, y, z], axis=-1)

    # Normals via numpy gradient (spacing = step_m).
    gy, gx = np.gradient(heightmap, step_m, step_m)  # gy = dh/dnorth_row, gx = dh/deast_col
    # row axis is +z (since row increases as north decreases).
    # So dh/dz = -gy (because north = radius - row*step, d/drow = -1/step, and gy is d/drow)
    # np.gradient gives spacing-normalized derivative: gy = dh/dnorth along row axis but
    # rows go from north+ to north-, so dh/drow*step = -dh/dnorth ⇒ dh/dnorth = -gy.
    # Therefore dh/dz (z = -north) = -dh/dnorth = gy.
    nx = -gx
    ny = np.ones_like(gx)
    nz = -gy
    norms = np.stack([nx, ny, nz], axis=-1).astype(np.float32).reshape(-1, 3)
    norms /= np.linalg.norm(norms, axis=1, keepdims=True).clip(min=1e-8)

    # Triangle indices (two tris per quad, CCW when viewed from +y).
    g = grid
    r = np.arange(g - 1, dtype=np.uint32)[:, None]
    c = np.arange(g - 1, dtype=np.uint32)[None, :]
    i0 = r * g + c
    i1 = (r + 1) * g + c
    i2 = (r + 1) * g + (c + 1)
    i3 = r * g + (c + 1)
    tri1 = np.stack([i0, i1, i2], axis=-1)
    tri2 = np.stack([i0, i2, i3], axis=-1)
    indices = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0).ravel()

    sampler = TerrainSampler(heightmap=heightmap, radius_m=radius_m)
    return TerrainData(vertices=vertices, normals=norms, indices=indices,
                       heightmap=heightmap, radius_m=radius_m, sampler=sampler,
                       origin_alt_m=h0)


def dump_obj(data: TerrainData, path) -> None:
    """Write the terrain mesh as a Wavefront .obj for eyeball inspection."""
    from pathlib import Path
    p = Path(path)
    with p.open("w") as f:
        for v in data.vertices:
            f.write(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for n in data.normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
        for i in range(0, len(data.indices), 3):
            a, b, c_ = data.indices[i] + 1, data.indices[i + 1] + 1, data.indices[i + 2] + 1
            f.write(f"f {a}//{a} {b}//{b} {c_}//{c_}\n")
