"""Download + cache CC0 3D mesh packs for the scene.

Currently we ship one pack — the Shapespark Low-Poly Plants Kit (CC0) —
whose monolithic glTF contains 30 plant meshes. It is fetched from the
project's GitHub release on first run and extracted into the cache; the
viewer then slices it into per-plant meshes at startup (see
:mod:`osm3denv.render.plants`).
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# Single pack today. ``directory`` is the cache subdir relative to textures_root
# parent (i.e. ``<cache_dir>/meshes/<id>``) and ``url`` is the stable release.
MESH_PACKS: dict[str, dict] = {
    "shapespark_plants": {
        "url": ("https://github.com/shapespark/shapespark-assets/releases/download/"
                "1.0.0/shapespark-low-poly-plants-kit.zip"),
        "directory": "shapespark-low-poly-plants-kit",
        # Path (relative to the extracted root) of the glTF asset Ogre will
        # load via its Assimp codec.
        "gltf": "shapespark-low-poly-plants-kit/shapespark-low-poly-plants-kit.gltf",
    },
}

_UA = "osm3denv/0.1 (+https://github.com/)"
_SENTINEL = ".downloaded"


def _pack_dir(cache_root: Path, pack_id: str) -> Path:
    return cache_root / "meshes" / pack_id


def pack_root(cache_root: Path, pack_id: str) -> Path:
    """Return the directory the pack extracts into (may or may not exist)."""
    info = MESH_PACKS[pack_id]
    return _pack_dir(cache_root, pack_id)


def pack_gltf_path(cache_root: Path, pack_id: str) -> Path | None:
    """Return the absolute path to the pack's main glTF, or None if missing."""
    info = MESH_PACKS[pack_id]
    p = _pack_dir(cache_root, pack_id) / info["gltf"]
    return p if p.is_file() else None


def ensure_pack(cache_root: Path, pack_id: str,
                *, timeout: float = 300.0) -> Path | None:
    """Download + extract if not cached. Returns the pack root dir or None."""
    info = MESH_PACKS[pack_id]
    d = _pack_dir(cache_root, pack_id)
    gltf = d / info["gltf"]
    if gltf.is_file():
        return d

    d.mkdir(parents=True, exist_ok=True)
    zip_path = d / f"{pack_id}.zip"
    log.info("downloading mesh pack %s (%s)", pack_id, info["url"])
    try:
        req = Request(info["url"], headers={"User-Agent": _UA})
        with urlopen(req, timeout=timeout) as resp:
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(d)
        zip_path.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to fetch %s: %s (trees will fall back to procedural)",
                    pack_id, e)
        return None
    if not gltf.is_file():
        log.warning("mesh pack %s extracted but %s missing", pack_id, info["gltf"])
        return None
    return d


def meshes_root(cache_root: Path) -> Path:
    p = cache_root / "meshes"
    p.mkdir(parents=True, exist_ok=True)
    return p
