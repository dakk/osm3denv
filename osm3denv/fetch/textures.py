"""Download and cache CC0 PBR texture packs from ambientCG.

Each pack is a zip of PNGs (Color / NormalGL / Roughness / AmbientOcclusion /
Displacement). We cache packs in ``<cache_dir>/textures/<pack_id>/`` and the
fragment shaders reference the unpacked filenames verbatim.

Licence: ambientCG assets are CC0 (public domain), no attribution required.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# Short name → ambientCG asset ID at 2K PNG resolution. Short name is what
# materials.py checks via _pack_available(); if the download for any pack
# fails, the shader path for that material falls back to procedural.
TEXTURE_PACKS: dict[str, str] = {
    "asphalt": "Asphalt026A_2K-PNG",
    "brick":   "Bricks097_2K-PNG",
    "brick2":  "Bricks075A_2K-PNG",        # weathered older brick
    "brick3":  "Bricks031_2K-PNG",         # lighter cream brick
    "grass":   "Grass004_2K-PNG",
    "rock":    "Rock030_2K-PNG",
    "roof":    "RoofingTiles010_2K-PNG",
    "roof2":   "RoofingTiles002_2K-PNG",   # darker slate-like tiles
    "sand":    "Sand006_2K-PNG",
    "paved":   "PavingStones012_2K-PNG",   # residential / commercial / industrial
    "soil":    "Ground054_2K-PNG",         # farmland / dry earth
}

_UA = "osm3denv/0.1 (+https://github.com/)"
_URL = "https://ambientcg.com/get?file={pack}.zip"
_SUFFIXES = (".png", ".jpg", ".jpeg")


def _pack_dir(cache_root: Path, pack_id: str) -> Path:
    return cache_root / "textures" / pack_id


def pack_files(cache_root: Path, pack_id: str) -> dict[str, Path]:
    """Return a dict of {'color', 'normal', 'roughness', 'ao'} → file path.

    Empty dict means the pack hasn't been cached yet.
    """
    d = _pack_dir(cache_root, pack_id)
    if not d.is_dir():
        return {}
    out: dict[str, Path] = {}
    for p in d.iterdir():
        if p.suffix.lower() not in _SUFFIXES:
            continue
        low = p.name.lower()
        if "color" in low:
            out["color"] = p
        elif "normalgl" in low:
            out["normal"] = p
        elif "roughness" in low:
            out["roughness"] = p
        elif "ambientocclusion" in low:
            out["ao"] = p
    return out


def ensure_pack(cache_root: Path, pack_id: str,
                *, timeout: float = 60.0) -> dict[str, Path]:
    existing = pack_files(cache_root, pack_id)
    if existing:
        return existing

    d = _pack_dir(cache_root, pack_id)
    d.mkdir(parents=True, exist_ok=True)
    url = _URL.format(pack=pack_id)
    zip_path = d / f"{pack_id}.zip"

    log.info("downloading PBR pack %s (%s)", pack_id, url)
    try:
        req = Request(url, headers={"User-Agent": _UA})
        with urlopen(req, timeout=timeout) as resp:
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.lower().endswith(_SUFFIXES):
                    # Flatten: ignore any subdirectory the zip may contain.
                    target = d / Path(member).name
                    with zf.open(member) as src, target.open("wb") as dst:
                        dst.write(src.read())
        zip_path.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to fetch %s: %s (shaders will fall back to procedural)",
                    pack_id, e)
        return {}
    return pack_files(cache_root, pack_id)


def ensure_all(cache_root: Path) -> dict[str, dict[str, Path]]:
    """Download every pack in :data:`TEXTURE_PACKS` that isn't cached yet.

    Returns a mapping short_name → {'color', 'normal', 'roughness', 'ao'}.
    Missing packs appear as empty dicts; callers should treat those as a
    signal to use the procedural fallback material.
    """
    return {name: ensure_pack(cache_root, pid)
            for name, pid in TEXTURE_PACKS.items()}


def textures_root(cache_root: Path) -> Path:
    """Return the directory Ogre should add as a resource location."""
    p = cache_root / "textures"
    p.mkdir(parents=True, exist_ok=True)
    return p
