from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TerrainSampler:
    """Bilinear height lookup over the ENU terrain grid.

    ``heightmap`` is arranged with row 0 at north_max (+radius) and col 0 at
    east_min (−radius); (east, north) queries outside the grid are clamped to
    the nearest edge.
    """

    heightmap: np.ndarray  # shape (G, G), float32, meters
    radius_m: float

    def height_at(self, east, north):
        g = self.heightmap.shape[0]
        dx = 2.0 * self.radius_m / (g - 1)
        east_a = np.asarray(east, dtype=np.float64)
        north_a = np.asarray(north, dtype=np.float64)
        col = np.clip((east_a + self.radius_m) / dx, 0.0, g - 1)
        row = np.clip((self.radius_m - north_a) / dx, 0.0, g - 1)
        r0 = np.floor(row).astype(int); r1 = np.clip(r0 + 1, 0, g - 1)
        c0 = np.floor(col).astype(int); c1 = np.clip(c0 + 1, 0, g - 1)
        dr = (row - r0).astype(np.float32); dc = (col - c0).astype(np.float32)
        v00 = self.heightmap[r0, c0]; v01 = self.heightmap[r0, c1]
        v10 = self.heightmap[r1, c0]; v11 = self.heightmap[r1, c1]
        top = v00 * (1 - dc) + v01 * dc
        bot = v10 * (1 - dc) + v11 * dc
        out = top * (1 - dr) + bot * dr
        if np.isscalar(east) and np.isscalar(north):
            return float(out)
        return out
