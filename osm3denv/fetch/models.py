"""Locate evolveduk tree/plant GLB models from the user's cache directory.

Models are CC BY 4.0 — credit: evolveduk (https://sketchfab.com/evolveduk/models).
Download each GLB from Sketchfab and place it as::

    <cache_dir>/evolveduk/<slug>.glb
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# OSM vegetation type → list of slugs (randomly sampled per instance)
MODEL_POOL: dict[str, list[str]] = {
    "forest":        ["fir_tree", "spruce", "pine_tree", "birch_tree",
                      "maple_tree", "old_tree"],
    "park":          ["oak_tree", "beech_tree", "elm_tree", "willow",
                      "maple_tree", "old_tree"],
    "orchard":       ["acacia_tree", "oak_tree", "beech_tree", "maple_tree"],
    "scrub":         ["juniper", "fern", "grass", "grass_patches", "dry_grass", "grass_claster"],
    "heath":         ["juniper", "fern", "daisy", "daisies", "grass", "grass_patches", "dry_grass", "grass_claster"],
    # ground cover base layer applied to every vegetation polygon
    "_ground_cover": ["grass_claster", "grass_claster", "grass_claster",
                      "grass_claster", "grass_claster", "grass_claster",
                      "grass_claster", "grass_claster", "grass_claster",
                      "daisy", "daisies"],
    "cemetery":      ["juniper"],
    "garden":        ["willow", "beech_tree", "oak_tree", "fern", "daisy", "daisies",
                      "grass", "grass_patches"],
    "village_green": ["oak_tree", "elm_tree", "willow", "maple_tree"],
    "allotments":    ["acacia_tree", "oak_tree", "maple_tree"],
    "residential":   ["oak_tree", "beech_tree", "willow", "maple_tree",
                      "elm_tree", "birch_tree", "acacia_tree"],
    "tree":          ["oak_tree", "elm_tree", "beech_tree", "birch_tree",
                      "willow", "pine_tree", "fir_tree", "acacia_tree",
                      "maple_tree", "old_tree"],
}

# Known download URLs — models without a URL were sourced manually
_DOWNLOAD_URLS: dict[str, str] = {
    "juniper":     "https://sketchfab.com/3d-models/juniper-e74c7156984542c2a4cdf6ce8a0c8e96",
    "birch_tree":  "https://sketchfab.com/3d-models/birch-tree-aa842dffd9654d33b8b91170ce83c172",
    "fir_tree":    "https://sketchfab.com/3d-models/fir-tree-3f39aa5485e94477a36b435f7a1a8b54",
    "oak_tree":    "https://sketchfab.com/3d-models/oak-tree-6468dd4d3eb240ef902b9057d9913606",
    "elm_tree":    "https://sketchfab.com/3d-models/elm-tree-f36443d3e59946bfa38b3f5cb90ea5fa",
    "acacia_tree": "https://sketchfab.com/3d-models/acacia-tree-bb14c2bb679b4d0bb1c578a27e2ddabf",
    "beech_tree":  "https://sketchfab.com/3d-models/beech-tree-0983d8933531491f9be71c669e8a907b",
    "pine_tree":   "https://sketchfab.com/3d-models/pine-tree-d45218a3fab349e5b1de040f29e7b6f9",
    "willow":      "https://sketchfab.com/3d-models/willow-422de2372f3d46dfb314a0cd5da512fe",
    "spruce":      "https://sketchfab.com/3d-models/spruce-a50a5df3164246a5af97992cec33a143",
    "fern":        "https://sketchfab.com/evolveduk/models",
    "daisy":       "https://sketchfab.com/evolveduk/models",
    "daisies":     "https://sketchfab.com/evolveduk/models",
    "maple_tree":  "https://sketchfab.com/evolveduk/models",
    "old_tree":    "https://sketchfab.com/evolveduk/models",
    "bamboo":        "https://sketchfab.com/evolveduk/models",
    "grass":         "https://sketchfab.com/3d-models/grass-674d42354f6348a7a85fc06a7f004db5",
    "grass_patches": "https://sketchfab.com/3d-models/grass-patches-6952780b80594a31aab2dedf7249a47a",
    "dry_grass":     "https://sketchfab.com/3d-models/dry-grass-d1484537470441b999d37077aeb9b47a",
    "grass_claster": "https://sketchfab.com/3d-models/grass-claster-downoad-like-please-832eb6c9c5b24790b1ca24ad7dfdcdba",
    "curly_palm":  "https://sketchfab.com/evolveduk/models",
    "date_palm":   "https://sketchfab.com/evolveduk/models",
}

# Per-model height override (h_min, h_max) in metres.
# When set, the instance height is drawn from this range instead of the
# VegType range, so ground cover plants don't scale up to tree size.
MODEL_HEIGHTS: dict[str, tuple[float, float]] = {
    "daisy":   (0.1, 0.3),
    "daisies": (0.1, 0.3),
    "fern":          (0.2, 0.6),
    "bamboo":        (4.0, 10.0),
    "grass":         (0.1, 0.4),
    "grass_patches": (0.1, 0.3),
    "dry_grass":     (0.1, 0.4),
    "grass_claster": (0.1, 0.4),
}

_ALL_SLUGS: list[str] = sorted(_DOWNLOAD_URLS.keys())
SUBDIR = "nature"

# ---------------------------------------------------------------------------
# Street furniture models
# ---------------------------------------------------------------------------

FURNITURE_MODELS: dict[str, str] = {
    "street_lamp": "https://sketchfab.com/3d-models/street-lamp-152055979ddd48669529f5d4f5f3543c#download",
}
FURNITURE_SUBDIR = "streetfurniture"


def fetch_furniture(cache_dir: Path) -> dict[str, Path]:
    """Return ``{slug: glb_path}`` for each GLB found in ``<cache_dir>/streetfurniture/``.

    Missing files are logged as warnings with their download URLs.
    """
    dest = cache_dir / FURNITURE_SUBDIR
    dest.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    missing: list[str] = []

    for slug, url in FURNITURE_MODELS.items():
        p = dest / f"{slug}.glb"
        if p.exists():
            result[slug] = p
        else:
            missing.append((slug, url))

    if missing:
        lines = "\n  ".join(f"{s}.glb  →  {u}" for s, u in missing)
        log.warning(
            "street furniture models missing from %s\n"
            "Download as GLB from Sketchfab and rename to <slug>.glb:\n  %s",
            dest, lines,
        )

    return result


def fetch(cache_dir: Path, slugs: list[str] | None = None) -> dict[str, Path]:
    """Return ``{slug: glb_path}`` for each GLB found in ``<cache_dir>/evolveduk/``.

    Missing files are logged as warnings with their download URLs.
    """
    dest = cache_dir / SUBDIR
    dest.mkdir(parents=True, exist_ok=True)

    if slugs is None:
        slugs = _ALL_SLUGS

    result: dict[str, Path] = {}
    missing: list[str] = []

    for slug in slugs:
        p = dest / f"{slug}.glb"
        if p.exists():
            result[slug] = p
        else:
            missing.append(slug)

    if missing:
        lines = "\n  ".join(
            f"{s}.glb  →  {_DOWNLOAD_URLS.get(s, '?')}" for s in missing
        )
        log.warning(
            "evolveduk models missing from %s\n"
            "Download as GLB from Sketchfab (CC BY 4.0 — credit: evolveduk) "
            "and rename to <slug>.glb:\n  %s",
            dest, lines,
        )

    return result
