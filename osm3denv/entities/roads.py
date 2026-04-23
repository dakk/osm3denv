"""Roads entity — rasterises path/track ways into a terrain splatmap.

Road geometry is painted into a 2-D luminance texture that the terrain shader
samples to blend a dirt colour over the landscape.  This avoids separate road
meshes, alpha-sorting artifacts, and depth-fighting entirely.
"""
from __future__ import annotations

import logging

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_HALF_WIDTH: dict[str, float] = {
    "track":      2.0,
    "path":       0.5,
    "footway":    0.5,
    "bridleway":  1.0,
    "cycleway":   0.6,
    "pedestrian": 1.5,
}

_SMAP_RES    = 2048   # splatmap resolution (pixels per side)
_MIN_HALF_PX = 0.8    # minimum half-width in splatmap pixels (ensures sub-metre ways are visible)


def _rasterize_segment(smap: np.ndarray,
                       x0: float, y0: float,
                       x1: float, y1: float,
                       half_w: float) -> None:
    """Paint a thick line segment into *smap* with a smoothstep distance falloff.

    Using a soft falloff (instead of a hard binary mask) naturally anti-aliases
    diagonal segments and eliminates staircase artifacts without heavy blur.
    """
    H, W = smap.shape
    # Feather zone extends half_w outward for a smooth edge.
    feather = max(1.5, half_w * 0.5)
    outer   = half_w + feather

    x_lo = max(0, int(min(x0, x1) - outer - 1))
    x_hi = min(W - 1, int(max(x0, x1) + outer + 1))
    y_lo = max(0, int(min(y0, y1) - outer - 1))
    y_hi = min(H - 1, int(max(y0, y1) + outer + 1))
    if x_hi < x_lo or y_hi < y_lo:
        return

    px = np.arange(x_lo, x_hi + 1, dtype=np.float32)
    py = np.arange(y_lo, y_hi + 1, dtype=np.float32)
    gx, gy = np.meshgrid(px, py)

    dx, dy = x1 - x0, y1 - y0
    seg_len_sq = float(dx * dx + dy * dy)
    if seg_len_sq > 1e-12:
        t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / seg_len_sq, 0.0, 1.0)
        cx = x0 + t * dx
        cy = y0 + t * dy
    else:
        cx, cy = float(x0), float(y0)

    dist  = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)
    inner = half_w - feather
    # smoothstep from inner (fully opaque) to outer (transparent)
    s = np.clip((dist - inner) / (2.0 * feather), 0.0, 1.0)
    val = 1.0 - s * s * (3.0 - 2.0 * s)

    tile = smap[y_lo:y_hi + 1, x_lo:x_hi + 1]
    np.maximum(tile, val, out=tile)


class Roads(MapEntity):
    """Dirt tracks and paths baked into the terrain via a splatmap texture."""

    def __init__(self, osm: OSMData, frame: Frame, radius_m: float) -> None:
        self._osm = osm
        self._frame = frame
        self._radius_m = radius_m
        self._splatmap: np.ndarray | None = None

    def build(self) -> None:
        r   = float(self._radius_m)
        res = _SMAP_RES
        # Row 0 = south (north = -r), row res-1 = north (north = +r)
        smap = np.zeros((res, res), dtype=np.float32)
        n_ways = 0

        for way in self._osm.filter_ways(lambda t: t.get("highway") in _HALF_WIDTH):
            geom = way.geometry
            if len(geom) < 2:
                continue

            lons = np.fromiter((p[0] for p in geom), np.float64, count=len(geom))
            lats = np.fromiter((p[1] for p in geom), np.float64, count=len(geom))
            east, north = self._frame.to_enu(lons, lats)

            if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                continue

            # ENU → pixel: x increases eastward, y increases northward (row 0 = south)
            px = (east  + r) / (2.0 * r) * res
            py = (north + r) / (2.0 * r) * res

            half_w_m  = _HALF_WIDTH[way.tags["highway"]]
            half_w_px = max(_MIN_HALF_PX, half_w_m / (2.0 * r) * res)

            for i in range(len(px) - 1):
                _rasterize_segment(smap, px[i], py[i], px[i + 1], py[i + 1], half_w_px)
            n_ways += 1

        try:
            from scipy.ndimage import gaussian_filter
            smap = gaussian_filter(smap, sigma=0.8)
        except ImportError:
            pass

        self._splatmap = smap.clip(0.0, 1.0)
        log.info("roads: %d ways → %dx%d splatmap", n_ways, res, res)

    def attach_to(self, parent) -> None:
        if self._splatmap is None:
            return
        from panda3d.core import Texture

        smap = self._splatmap
        res  = smap.shape[0]
        data = (smap * 255.0).clip(0, 255).astype(np.uint8)

        tex = Texture("road_splatmap")
        tex.setup2dTexture(res, res, Texture.T_unsigned_byte, Texture.F_luminance)
        tex.setRamImage(memoryview(data))

        # Override the blank default set by terrain on the terrain node
        terrain_np = parent.find("**/terrain")
        if not terrain_np.isEmpty():
            terrain_np.setShaderInput("u_road_splatmap", tex)
        else:
            parent.setShaderInput("u_road_splatmap", tex)
