"""Texture loading from Poly Haven CDN (CC0) with local disk cache.

All textures are CC0 / public domain from https://polyhaven.com — no attribution
required.  Textures are downloaded on first use and cached in
~/.cache/procbuilding/textures/ so subsequent runs are instant.

If the download fails (offline, etc.) a tiny solid-color fallback is returned
so the application keeps running.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from panda3d.core import PNMImage, Texture

# ---------------------------------------------------------------------------
# Poly Haven diffuse map URL pattern (1 K JPEGs, CC0)
# ---------------------------------------------------------------------------
_PH_BASE = "https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k"

# kind → Poly Haven asset slug
_SLUG: dict[str, str] = {
    # plaster variants
    "plaster":          "white_plaster_02",
    "plaster_painted":  "painted_concrete_02",
    # roof variants
    "roof_tile":        "clay_roof_tiles_02",
    "roof_slate":       "roof_slates_02",
    "roof_corrugated":  "corrugated_iron_02",
    # wood/door variants
    "wood":             "wood_cabinet_worn_long",
    "wood_planks":      "old_planks_02",
    # fixed materials (no variants)
    "concrete":         "concrete_floor_02",
    "metal":            "metal_plate",
    "brick":            "large_red_bricks",
    "cobblestone":      "cobblestone_floor_02",
}

_CACHE_DIR = Path.home() / ".cache" / "procbuilding" / "textures"
_cache_diff: dict[str, Texture] = {}
_cache_nor:  dict[str, Texture] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_texture(kind: str) -> Texture:
    """Return a cached diffuse Texture for *kind*, downloading if needed."""
    if kind not in _cache_diff:
        _cache_diff[kind] = _load_diff(kind)
    return _cache_diff[kind]


def get_normal_texture(kind: str) -> Texture:
    """Return a cached normal-map Texture for *kind*, downloading if needed."""
    if kind not in _cache_nor:
        _cache_nor[kind] = _load_nor(kind)
    return _cache_nor[kind]


# ---------------------------------------------------------------------------
# Internals — diffuse
# ---------------------------------------------------------------------------

def _load_diff(kind: str) -> Texture:
    slug = _SLUG[kind]
    local = _ensure_downloaded(slug, f"{slug}_diff_1k.jpg",
                               f"{_PH_BASE}/{slug}/{slug}_diff_1k.jpg", kind)
    if local is None:
        return _fallback_diff(kind)

    tex = Texture(kind)
    if not tex.read(local):
        return _fallback_diff(kind)

    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    # Poly Haven diffuse JPEGs are sRGB — tell the GPU to linearise on sample
    tex.setFormat(Texture.F_srgb)
    return tex


_FALLBACK_RGB: dict[str, tuple[float, float, float]] = {
    "plaster":          (0.88, 0.84, 0.78),
    "plaster_painted":  (0.70, 0.68, 0.65),
    "roof_tile":        (0.46, 0.20, 0.12),
    "roof_slate":       (0.35, 0.35, 0.40),
    "roof_corrugated":  (0.55, 0.55, 0.55),
    "wood":             (0.55, 0.35, 0.15),
    "wood_planks":      (0.45, 0.30, 0.12),
    "concrete":         (0.50, 0.50, 0.50),
    "metal":            (0.62, 0.62, 0.64),
    "brick":            (0.68, 0.28, 0.18),
    "cobblestone":      (0.45, 0.42, 0.38),
}


def _fallback_diff(kind: str) -> Texture:
    r, g, b = _FALLBACK_RGB.get(kind, (0.5, 0.5, 0.5))
    img = PNMImage(4, 4, 3)
    for y in range(4):
        for x in range(4):
            img.setXel(x, y, r, g, b)
    tex = Texture(kind)
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    # Fallback colours are already linear — do NOT mark as sRGB
    return tex


# ---------------------------------------------------------------------------
# Internals — normal maps
# ---------------------------------------------------------------------------

def _load_nor(kind: str) -> Texture:
    slug = _SLUG[kind]
    local = _ensure_downloaded(slug, f"{slug}_nor_gl_1k.jpg",
                               f"{_PH_BASE}/{slug}/{slug}_nor_gl_1k.jpg",
                               f"{kind}_nor")
    if local is None:
        return _fallback_nor()

    tex = Texture(f"{kind}_nor")
    if not tex.read(local):
        return _fallback_nor()

    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setMagfilter(Texture.FTLinear)
    # Normal maps are linear data — do NOT mark as sRGB
    return tex


def _fallback_nor() -> Texture:
    """Flat normal map: RGB = (0.498, 0.498, 1.0) → tangent-space (0, 0, 1)."""
    img = PNMImage(4, 4, 3)
    for y in range(4):
        for x in range(4):
            img.setXel(x, y, 0.498, 0.498, 1.0)
    tex = Texture("fallback_nor")
    tex.load(img)
    tex.setWrapU(Texture.WMRepeat)
    tex.setWrapV(Texture.WMRepeat)
    return tex


# ---------------------------------------------------------------------------
# Shared download helper
# ---------------------------------------------------------------------------

def _ensure_downloaded(slug: str, filename: str, url: str, label: str):
    """Download *url* to cache if not present. Returns local Path or None."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = _CACHE_DIR / filename
    if not local.exists():
        print(f"  [textures] downloading '{label}' …", flush=True)
        try:
            urllib.request.urlretrieve(url, str(local))
        except Exception as exc:
            print(f"  [textures] download failed ({exc}); using fallback")
            return None
    return local
