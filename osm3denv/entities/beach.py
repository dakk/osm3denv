"""Beach entity — rasterises natural=beach polygons into a terrain splatmap.

Works identically to Roads: the splatmap texture is written as a shader input
on the terrain node so the terrain fragment shader can blend to pure sand in
tagged beach areas.
"""
from __future__ import annotations

import logging

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_SMAP_RES = 2048


def _points_in_polygon(pts_x: np.ndarray, pts_y: np.ndarray,
                       poly_x: np.ndarray, poly_y: np.ndarray) -> np.ndarray:
    inside = np.zeros(len(pts_x), dtype=bool)
    j = len(poly_x) - 1
    for i in range(len(poly_x)):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        crosses = (yi > pts_y) != (yj > pts_y)
        x_int   = (xj - xi) * (pts_y - yi) / ((yj - yi) + 1e-12) + xi
        inside ^= crosses & (pts_x < x_int)
        j = i
    return inside


def _rasterize_polygon(smap: np.ndarray,
                       px: np.ndarray, py: np.ndarray) -> None:
    """Fill the interior of a pixel-space polygon into *smap*."""
    H, W = smap.shape
    x_lo = max(0, int(px.min()))
    x_hi = min(W - 1, int(px.max()) + 1)
    y_lo = max(0, int(py.min()))
    y_hi = min(H - 1, int(py.max()) + 1)
    if x_hi <= x_lo or y_hi <= y_lo:
        return

    xs = np.arange(x_lo, x_hi + 1, dtype=np.float32)
    ys = np.arange(y_lo, y_hi + 1, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    pts_x = gx.ravel()
    pts_y = gy.ravel()

    inside = _points_in_polygon(pts_x, pts_y, px, py)

    ix = pts_x[inside].astype(np.int32)
    iy = pts_y[inside].astype(np.int32)
    smap[iy, ix] = 1.0


class Beach(MapEntity):
    """Natural beach polygons baked into the terrain via a splatmap."""

    def __init__(self, osm: OSMData, frame: Frame, radius_m: float) -> None:
        self._osm      = osm
        self._frame    = frame
        self._radius_m = radius_m
        self._splatmap: np.ndarray | None = None

    def _to_px(self, east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r   = float(self._radius_m)
        res = _SMAP_RES
        px  = (east  + r) / (2.0 * r) * res
        py  = (north + r) / (2.0 * r) * res
        return px.astype(np.float32), py.astype(np.float32)

    def _process_ring(self, smap: np.ndarray, geom: list) -> bool:
        r    = float(self._radius_m)
        lons = np.fromiter((p[0] for p in geom), np.float64, len(geom))
        lats = np.fromiter((p[1] for p in geom), np.float64, len(geom))
        east, north = self._frame.to_enu(lons, lats)
        if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
            return False
        px, py = self._to_px(east, north)
        _rasterize_polygon(smap, px, py)
        return True

    def build(self) -> None:
        smap    = np.zeros((_SMAP_RES, _SMAP_RES), dtype=np.float32)
        n_polys = 0

        for way in self._osm.filter_ways(lambda t: t.get("natural") == "beach"):
            if len(way.geometry) >= 3:
                if self._process_ring(smap, way.geometry):
                    n_polys += 1

        for rel in self._osm.filter_relations(lambda t: t.get("natural") == "beach"):
            for role, ring in rel.rings:
                if role in ("outer", "") and len(ring) >= 3:
                    if self._process_ring(smap, ring):
                        n_polys += 1

        try:
            from scipy.ndimage import gaussian_filter
            smap = gaussian_filter(smap, sigma=1.2)
        except ImportError:
            pass

        self._splatmap = smap.clip(0.0, 1.0)
        log.info("beach: %d polygons → %dx%d splatmap", n_polys, _SMAP_RES, _SMAP_RES)

    def attach_to(self, parent) -> None:
        if self._splatmap is None:
            return
        from panda3d.core import Texture

        data = (self._splatmap * 255.0).clip(0, 255).astype(np.uint8)
        log.info("beach splatmap: max=%.1f nonzero=%d",
                 float(data.max()), int(np.count_nonzero(data)))
        tex  = Texture("beach_splatmap")
        tex.setup2dTexture(_SMAP_RES, _SMAP_RES,
                           Texture.T_unsigned_byte, Texture.F_luminance)
        tex.setRamImage(data.tobytes())

        terrain_np = parent.find("**/terrain")
        log.info("beach splatmap: terrain node found=%s", not terrain_np.isEmpty())
        target = terrain_np if not terrain_np.isEmpty() else parent
        target.setShaderInput("u_beach_splatmap", tex)
