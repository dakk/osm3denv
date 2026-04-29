"""Download and cache PBR terrain textures from ambientCG (CC0 licence).

Each asset is a 1K-JPG ZIP from https://ambientcg.com containing at least
``<ID>_1K_Color.jpg`` (diffuse) and ``<ID>_1K_NormalGL.jpg`` (OpenGL-convention
normal map, green-up).  Files are extracted once and cached under
``cache_dir/<asset_id>/``.  A missing or failed download returns an empty dict
for that type so callers can fall back to solid-colour placeholders.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_BASE_URL   = "https://ambientcg.com/get?file={asset_id}_1K-JPG.zip"
_USER_AGENT = "osm3denv/0.2 (+https://github.com/local/osm3denv)"

# ambientCG asset IDs — all CC0.
TERRAIN_ASSETS: dict[str, str] = {
    "sand":       "Ground026",   # fine inland / desert sand
    "grass":      "Ground037",   # grass with soil variation
    "rock":       "Rock022",     # rock face / cliff
    "beach_sand": "Ground054",   # coarse wet beach sand
}

ROAD_ASSETS: dict[str, str] = {
    "road":  "Road007",          # asphalt road surface (CC0)
    "path":  "PavingStones070",  # paved path / footway (CC0)
}

BUILDING_ASSETS: dict[str, str] = {
    "brick":      "Bricks026",         # red-orange clay brick (rare)
    "plaster":    "Plaster001",        # cream painted plaster
    "plaster_b":  "Plaster002",        # warm off-white plaster
    "plaster_c":      "Plaster004",          # cool-tinted light plaster
    "painted_plaster":  "PaintedPlaster004", # painted plaster variant A
    "painted_plaster_b":"PaintedPlaster012", # painted plaster variant B
    "painted_plaster_c":"PaintedPlaster017", # painted plaster variant C
    "concrete":   "Concrete025",       # exposed grey concrete
    "roof_tiles": "RoofingTiles012B",  # roof tiles
}


def _asset_paths(dest_dir: Path) -> dict[str, Path] | None:
    """Return {color, normal} if both files already exist in *dest_dir*."""
    color = normal = None
    if dest_dir.exists():
        for f in dest_dir.iterdir():
            lower = f.name.lower()
            if "_color" in lower and f.suffix in (".jpg", ".png"):
                color = f
            if "_normalgl" in lower and f.suffix in (".jpg", ".png"):
                normal = f
    if color and normal:
        return {"color": color, "normal": normal}
    return None


def _download_asset(asset_id: str, dest_dir: Path) -> dict[str, Path] | None:
    """Download one ambientCG ZIP, extract colour + normal, return paths."""
    cached = _asset_paths(dest_dir)
    if cached:
        log.debug("texture cache hit: %s", asset_id)
        return cached

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = _BASE_URL.format(asset_id=asset_id)
    log.info("downloading texture %s …", asset_id)
    try:
        r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=60)
        r.raise_for_status()
    except requests.RequestException as exc:
        log.warning("texture download failed (%s): %s", asset_id, exc)
        return None

    color = normal = None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                if "_color" in lower or "_normalgl" in lower:
                    data = zf.read(name)
                    out  = dest_dir / Path(name).name
                    out.write_bytes(data)
                    if "_color" in lower:
                        color = out
                    else:
                        normal = out
    except zipfile.BadZipFile as exc:
        log.warning("bad ZIP for %s: %s", asset_id, exc)
        return None

    if not color or not normal:
        log.warning("Color/NormalGL not found in ZIP for %s", asset_id)
        return None

    log.info("texture ready: %s", asset_id)
    return {"color": color, "normal": normal}


def fetch(cache_dir: Path) -> dict[str, dict[str, Path]]:
    """Download (once) and return ``{type: {color, normal}}`` for all terrain types."""
    result: dict[str, dict[str, Path]] = {}
    for tex_type, asset_id in TERRAIN_ASSETS.items():
        paths = _download_asset(asset_id, cache_dir / asset_id)
        result[tex_type] = paths or {}
    return result


def fetch_road(cache_dir: Path) -> dict[str, dict[str, Path]]:
    """Download (once) and return ``{type: {color, normal}}`` for road textures."""
    result: dict[str, dict[str, Path]] = {}
    for tex_type, asset_id in ROAD_ASSETS.items():
        paths = _download_asset(asset_id, cache_dir / asset_id)
        result[tex_type] = paths or {}
    return result


def fetch_building(cache_dir: Path) -> dict[str, dict[str, Path]]:
    """Download (once) and return ``{type: {color, normal}}`` for building textures."""
    result: dict[str, dict[str, Path]] = {}
    for tex_type, asset_id in BUILDING_ASSETS.items():
        paths = _download_asset(asset_id, cache_dir / asset_id)
        result[tex_type] = paths or {}
    return result
