"""High-resolution terrain tiles from AWS elevation-tiles-prod (Terrarium format).

Terrarium PNG tiles encode height as:
    height_m = R * 256 + G + B / 256 - 32768

Tiles are Web-Mercator ({z}/{x}/{y}) at any zoom level.  Resolution at the
equator is ~9 m/px at zoom 14 and ~4.5 m/px at zoom 15 — roughly 3-7× finer
than the SRTM1 HGT tiles used by srtm.py (~30 m).  At higher latitudes the
east-west pixel size shrinks (cos-latitude factor) while north-south stays the
same, so the effective resolution only improves toward the poles.

No authentication required; the same AWS S3 bucket already used for SRTM.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import requests

log = logging.getLogger(__name__)

_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_PX = 256   # pixels per tile side


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _lon_to_tx(lon: float, zoom: int) -> int:
    return int((lon + 180.0) / 360.0 * (1 << zoom))


def _lat_to_ty(lat: float, zoom: int) -> int:
    lat_r = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * (1 << zoom))


def _tx_to_lon(tx: int, zoom: int) -> float:
    """Longitude of the LEFT edge of tile tx."""
    return tx / (1 << zoom) * 360.0 - 180.0


def _ty_to_lat(ty: int, zoom: int) -> float:
    """Latitude of the TOP edge of tile ty."""
    n = math.pi - 2.0 * math.pi * ty / (1 << zoom)
    return math.degrees(math.atan(math.sinh(n)))


def zoom_for_step(step_m: float, lat: float) -> int:
    """Smallest zoom where tile resolution ≤ step_m (clamped to [10, 15]).

    Tiles oversample the terrain grid by ~1.5-3×, which is ideal for
    bilinear interpolation without fetching unnecessarily many tiles.
    """
    cos_lat = math.cos(math.radians(lat))
    for z in range(10, 16):
        res = 40_075_016.686 * cos_lat / ((1 << z) * _PX)
        if res <= step_m:
            return z
    return 15


# ---------------------------------------------------------------------------
# Tile I/O
# ---------------------------------------------------------------------------

def _download_tile(z: int, x: int, y: int, cache_dir: Path,
                   refresh: bool = False,
                   progress: str = "") -> Path | None:
    path = cache_dir / f"terrarium_{z}_{x}_{y}.png"
    if not refresh and path.exists() and path.stat().st_size > 0:
        log.debug("%sterrarium cache hit: %s", progress, path.name)
        return path
    url = _URL.format(z=z, x=x, y=y)
    log.info("%sterrarium downloading %s", progress, url)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404:
        log.warning("%sterrarium tile %d/%d/%d missing (ocean/nodata)", progress, z, x, y)
        return None
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _decode_tile(path: Path) -> np.ndarray:
    """Return (256, 256) float32 metres from a Terrarium PNG."""
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0 - 32768.0


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------

@dataclass
class TerrariumMosaic:
    """Stitched Terrarium tiles; bilinear sampling by (lat, lon)."""

    _data:  np.ndarray   # (H, W) float32 heights, row 0 = north
    _zoom:  int
    _gx0:   float        # global pixel x of mosaic column 0
    _gy0:   float        # global pixel y of mosaic row 0

    def _to_local(self, lat, lon):
        """Fractional (row, col) in the mosaic for scalar or array inputs."""
        n = float(1 << self._zoom)
        lat_r = np.radians(np.asarray(lat, dtype=np.float64))
        gx = (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0 * n * _PX
        gy = (1.0 - np.arcsinh(np.tan(lat_r)) / np.pi) / 2.0 * n * _PX
        return gy - self._gy0, gx - self._gx0

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinear height in metres; returns same shape as lat/lon inputs."""
        rows, cols = self._to_local(lat, lon)
        H, W = self._data.shape
        rows = np.clip(rows, 0.0, H - 1).astype(np.float32)
        cols = np.clip(cols, 0.0, W - 1).astype(np.float32)
        r0 = np.floor(rows).astype(np.intp)
        c0 = np.floor(cols).astype(np.intp)
        r1 = np.clip(r0 + 1, 0, H - 1)
        c1 = np.clip(c0 + 1, 0, W - 1)
        dr = rows - r0;  dc = cols - c0
        v = (self._data[r0, c0] * (1 - dc) + self._data[r0, c1] * dc) * (1 - dr) \
          + (self._data[r1, c0] * (1 - dc) + self._data[r1, c1] * dc) * dr
        return v.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mosaic(bbox: tuple[float, float, float, float], cache_dir: Path,
                zoom: int, refresh: bool = False) -> TerrariumMosaic:
    min_lon, min_lat, max_lon, max_lat = bbox

    tx_min = _lon_to_tx(min_lon, zoom)
    tx_max = _lon_to_tx(max_lon, zoom)
    ty_min = _lat_to_ty(max_lat, zoom)   # y grows southward
    ty_max = _lat_to_ty(min_lat, zoom)

    nx = tx_max - tx_min + 1
    ny = ty_max - ty_min + 1
    log.info("terrarium zoom=%d tiles=%dx%d (~%.0f m/px)",
             zoom, nx, ny,
             40_075_016.686 * math.cos(math.radians((min_lat + max_lat) / 2))
             / ((1 << zoom) * _PX))

    total = nx * ny
    mosaic = np.zeros((ny * _PX, nx * _PX), dtype=np.float32)
    for iy, ty in enumerate(range(ty_min, ty_max + 1)):
        for ix, tx in enumerate(range(tx_min, tx_max + 1)):
            n_done = iy * nx + ix + 1
            p = _download_tile(zoom, tx, ty, cache_dir, refresh=refresh,
                               progress=f"[{n_done}/{total}] ")
            if p is not None:
                mosaic[iy * _PX:(iy + 1) * _PX,
                       ix * _PX:(ix + 1) * _PX] = _decode_tile(p)

    # Global pixel origin of the top-left corner of the mosaic.
    n = float(1 << zoom)
    gx0 = tx_min * _PX
    top_lat_r = math.radians(_ty_to_lat(ty_min, zoom))
    gy0 = (1.0 - math.asinh(math.tan(top_lat_r)) / math.pi) / 2.0 * n * _PX

    return TerrariumMosaic(_data=mosaic, _zoom=zoom, _gx0=gx0, _gy0=gy0)


def loader(cache_dir: Path, zoom: int | None = None,
           refresh: bool = False) -> Callable:
    """Return a ``bbox → TerrariumMosaic`` callable.

    If *zoom* is ``None`` it is auto-selected per call from the bbox size so
    the tile resolution is roughly equal to the terrain grid step.
    """
    def _load(bbox: tuple[float, float, float, float]) -> TerrariumMosaic:
        min_lon, min_lat, max_lon, max_lat = bbox
        z = zoom if zoom is not None else zoom_for_step(
            step_m=_bbox_step_m(min_lat, max_lat, max_lon - min_lon),
            lat=(min_lat + max_lat) / 2.0,
        )
        return load_mosaic(bbox, cache_dir, zoom=z, refresh=refresh)
    return _load


def _bbox_step_m(min_lat: float, max_lat: float, lon_span: float) -> float:
    """Approximate ground pixel size for a bbox (used for auto-zoom)."""
    lat_m = (max_lat - min_lat) * 111_320.0
    mid_lat = (min_lat + max_lat) / 2.0
    lon_m = lon_span * 111_320.0 * math.cos(math.radians(mid_lat))
    return min(lat_m, lon_m) / 512.0   # target ~512 samples across the bbox
