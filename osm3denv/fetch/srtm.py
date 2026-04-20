"""SRTM 1 arc-second heightmap access via tilezen's public S3 bucket.

No authentication or API key required. Tiles are cached as uncompressed .hgt
files under ``cache_dir``. The :class:`HgtMosaic` returned by :func:`load_mosaic`
supports bilinear sampling across tile boundaries.
"""
from __future__ import annotations

import gzip
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import requests

log = logging.getLogger(__name__)

SKADI_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{dir}/{name}.hgt.gz"
TILE_SIZE = 3601  # 1 arc-second → 3600 steps + 1


def tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def tile_url(lat_floor: int, lon_floor: int) -> str:
    ns_dir = f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}"
    return SKADI_URL.format(dir=ns_dir, name=tile_name(lat_floor, lon_floor))


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    out = []
    for lat_f in range(math.floor(min_lat), math.floor(max_lat) + 1):
        for lon_f in range(math.floor(min_lon), math.floor(max_lon) + 1):
            out.append((lat_f, lon_f))
    return out


def _download_tile(lat_f: int, lon_f: int, cache_dir: Path,
                   refresh: bool = False) -> Path | None:
    name = tile_name(lat_f, lon_f)
    path = cache_dir / f"{name}.hgt"
    if not refresh and path.exists() and path.stat().st_size > 0:
        log.info("srtm cache hit: %s", path.name)
        return path
    url = tile_url(lat_f, lon_f)
    log.info("srtm downloading %s", url)
    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        log.warning("tile %s missing (ocean/nodata)", name)
        return None
    r.raise_for_status()
    raw = gzip.decompress(r.content)
    tmp = path.with_suffix(".hgt.tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return path


def _read_hgt(path: Path) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=">i2")
    side = int(math.sqrt(raw.size))
    if side * side != raw.size:
        raise ValueError(f"{path}: unexpected size {raw.size}")
    arr = raw.astype(np.float32).reshape((side, side))
    arr[arr < -1000] = np.nan  # SRTM void marker is -32768
    return arr


@dataclass
class HgtMosaic:
    """Holds the HGT tiles covering a bbox and samples bilinearly across them."""

    tiles: dict[tuple[int, int], np.ndarray | None]

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinear height in meters at scalar or array (lat, lon) in degrees.

        Missing tiles (ocean) and voids return 0.
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        out = np.zeros(lat.shape, dtype=np.float32) if lat.shape else np.zeros((), dtype=np.float32)

        lat_f = np.floor(lat).astype(int)
        lon_f = np.floor(lon).astype(int)
        for key in {(int(a), int(b)) for a, b in zip(np.ravel(lat_f), np.ravel(lon_f))}:
            arr = self.tiles.get(key)
            mask = (lat_f == key[0]) & (lon_f == key[1])
            if arr is None:
                continue
            side = arr.shape[0]
            row = (key[0] + 1 - lat[mask]) * (side - 1)
            col = (lon[mask] - key[1]) * (side - 1)
            row = np.clip(row, 0.0, side - 1)
            col = np.clip(col, 0.0, side - 1)
            r0 = np.floor(row).astype(int); r1 = np.clip(r0 + 1, 0, side - 1)
            c0 = np.floor(col).astype(int); c1 = np.clip(c0 + 1, 0, side - 1)
            dr = (row - r0).astype(np.float32); dc = (col - c0).astype(np.float32)
            v00 = arr[r0, c0]; v01 = arr[r0, c1]
            v10 = arr[r1, c0]; v11 = arr[r1, c1]
            # NaN fallback: use mean of non-NaN corners, else 0.
            stack = np.stack([v00, v01, v10, v11])
            with np.errstate(invalid="ignore"):
                fill = np.where(np.all(np.isnan(stack), axis=0), 0.0, np.nanmean(stack, axis=0))
            v00 = np.where(np.isnan(v00), fill, v00)
            v01 = np.where(np.isnan(v01), fill, v01)
            v10 = np.where(np.isnan(v10), fill, v10)
            v11 = np.where(np.isnan(v11), fill, v11)
            top = v00 * (1 - dc) + v01 * dc
            bot = v10 * (1 - dc) + v11 * dc
            out[mask] = top * (1 - dr) + bot * dr
        return out


def load_mosaic(bbox: tuple[float, float, float, float], cache_dir: Path,
                refresh: bool = False) -> HgtMosaic:
    tiles: dict[tuple[int, int], np.ndarray | None] = {}
    for lat_f, lon_f in tiles_for_bbox(bbox):
        p = _download_tile(lat_f, lon_f, cache_dir, refresh=refresh)
        tiles[(lat_f, lon_f)] = _read_hgt(p) if p is not None else None
    return HgtMosaic(tiles=tiles)


def loader(cache_dir: Path, refresh: bool = False) -> Callable[
        [tuple[float, float, float, float]], HgtMosaic]:
    def _load(bbox):
        return load_mosaic(bbox, cache_dir, refresh=refresh)
    return _load
