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
        """Exact y that the rendered terrain triangle has at (east, north).

        Matches mesh/terrain.py: each quad is split along the NW→SE diagonal
        into triangles (NW, SW, SE) and (NW, SE, NE). The hit triangle is
        chosen by whether v > u (SW side) or v ≤ u (NE side) within the cell.
        """
        g = self.heightmap.shape[0]
        step = 2.0 * self.radius_m / (g - 1)
        east_a = np.asarray(east, dtype=np.float64)
        north_a = np.asarray(north, dtype=np.float64)
        col = np.clip((east_a + self.radius_m) / step, 0.0, g - 1)
        row = np.clip((self.radius_m - north_a) / step, 0.0, g - 1)
        c0 = np.floor(col).astype(int); c1 = np.clip(c0 + 1, 0, g - 1)
        r0 = np.floor(row).astype(int); r1 = np.clip(r0 + 1, 0, g - 1)
        u = (col - c0).astype(np.float32)
        v = (row - r0).astype(np.float32)
        h_nw = self.heightmap[r0, c0]
        h_ne = self.heightmap[r0, c1]
        h_sw = self.heightmap[r1, c0]
        h_se = self.heightmap[r1, c1]
        # Triangle 1 (NW, SW, SE) if v > u: h = (1-v) h_nw + (v-u) h_sw + u h_se
        # Triangle 2 (NW, SE, NE) otherwise: h = (1-u) h_nw + (u-v) h_ne + v h_se
        in_tri1 = v > u
        t1 = (1 - v) * h_nw + (v - u) * h_sw + u * h_se
        t2 = (1 - u) * h_nw + (u - v) * h_ne + v * h_se
        out = np.where(in_tri1, t1, t2)
        if np.isscalar(east) and np.isscalar(north):
            return float(out)
        return out
