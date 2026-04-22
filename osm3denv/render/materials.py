"""Programmatic materials for each scene layer."""
from __future__ import annotations

from pathlib import Path

import Ogre

from osm3denv.fetch.textures import TEXTURE_PACKS

# Set once by the viewer at startup with the root of the cached texture packs
# (e.g. ~/.cache/osm3denv/textures/). ``roads()`` and friends use this to
# decide whether to return a PBR material or fall back to the procedural one.
_TEXTURE_ROOT: Path | None = None


def set_texture_root(root: Path | None) -> None:
    global _TEXTURE_ROOT
    _TEXTURE_ROOT = root


def _pack_available(short_name: str) -> bool:
    pack_id = TEXTURE_PACKS.get(short_name)
    if _TEXTURE_ROOT is None or pack_id is None:
        return False
    d = _TEXTURE_ROOT / pack_id
    if not d.is_dir():
        return False
    names = [p.name.lower() for p in d.iterdir()]
    # Color + NormalGL + Roughness are the minimum we need.
    return (any("color" in n for n in names)
            and any("normalgl" in n for n in names)
            and any("roughness" in n for n in names))


def _make(name: str, diffuse: tuple[float, float, float],
          *, specular: tuple[float, float, float] | None = None,
          alpha: float = 1.0, two_sided: bool = False,
          depth_bias: tuple[float, float] | None = None) -> str:
    mm = Ogre.MaterialManager.getSingleton()
    if mm.resourceExists(name, "General"):
        return name
    mat = mm.create(name, "General")
    pass_ = mat.getTechnique(0).getPass(0)
    pass_.setDiffuse(diffuse[0], diffuse[1], diffuse[2], alpha)
    pass_.setAmbient(diffuse[0], diffuse[1], diffuse[2])
    if specular is not None:
        pass_.setSpecular(specular[0], specular[1], specular[2], 1.0)
        pass_.setShininess(32.0)
    if alpha < 1.0:
        pass_.setSceneBlending(Ogre.SBT_TRANSPARENT_ALPHA)
        pass_.setDepthWriteEnabled(False)
    if two_sided:
        pass_.setCullingMode(Ogre.CULL_NONE)
    if depth_bias is not None:
        # Polygon-offset the decal toward the camera so it wins the depth test
        # against the terrain it sits on. (constant, slope) in depth-buffer units.
        pass_.setDepthBias(depth_bias[0], depth_bias[1])
    return name


def terrain() -> str:
    if all(_pack_available(p) for p in ("grass", "rock", "sand")):
        return "osm3d/terrain_pbr_full"
    if _pack_available("grass"):
        return "osm3d/terrain_pbr"
    return _make("osm3d/terrain", (0.45, 0.55, 0.35))


def buildings() -> str:
    if _pack_available("brick") and _pack_available("roof"):
        return "osm3d/buildings_pbr_full"
    if _pack_available("brick"):
        return "osm3d/buildings_pbr"
    return _make("osm3d/buildings", (0.80, 0.75, 0.65),
                 specular=(0.10, 0.10, 0.10))


# Each variant is (brick_pack, roof_pack, material_name). Dispatch picks the
# first variant whose packs are all cached; if none qualify, falls back to
# the default full/partial/procedural chain used by buildings().
_BUILDING_VARIANTS: list[tuple[str, str, str]] = [
    ("brick",  "roof",  "osm3d/buildings_pbr_full"),
    ("brick2", "roof",  "osm3d/buildings_pbr_v1"),
    ("brick3", "roof2", "osm3d/buildings_pbr_v2"),
]


def buildings_for_variant(variant: int) -> str:
    """Pick a building material for a deterministic per-way variant index.

    If the variant's PBR packs aren't all cached, cascade through the other
    variants, then fall back to :func:`buildings` so the scene is still
    rendered (just with less diversity).
    """
    n = len(_BUILDING_VARIANTS)
    for offset in range(n):
        b, r, name = _BUILDING_VARIANTS[(variant + offset) % n]
        if _pack_available(b) and _pack_available(r):
            return name
    return buildings()


def roads() -> str:
    if _pack_available("asphalt"):
        return "osm3d/roads_pbr"
    return _make("osm3d/roads", (0.22, 0.22, 0.22),
                 depth_bias=(10.0, 5.0))


def roads_for_kind(kind: str) -> str:
    """Pick the road material for an OSM way classification.

    ``kind`` is one of the values produced by mesh.roads._classify:
    asphalt_major, asphalt_minor, paved, dirt, rail. Falls back to the
    procedural ``osm3d/roads`` material if the required PBR pack isn't cached.
    """
    pack_for_kind = {
        "asphalt_major": "asphalt",
        "asphalt_minor": "asphalt",
        "paved":         "paved",
        "dirt":          "soil",
        "rail":          "rock",
        "sidewalk":      "paved",
    }
    pack = pack_for_kind.get(kind, "asphalt")
    if _pack_available(pack):
        return f"osm3d/roads/{kind}"
    # Fallback: use the legacy procedural road material for any kind we can't
    # texture right now (keeps dirt paths, rail etc. visible instead of black).
    return _make("osm3d/roads", (0.22, 0.22, 0.22),
                 depth_bias=(10.0, 5.0))


def water() -> str:
    return _make("osm3d/water", (0.20, 0.35, 0.55), alpha=0.85,
                 depth_bias=(1.0, 1.0))


def vegetation() -> str:
    if _pack_available("grass"):
        return "osm3d/vegetation_pbr"
    return _make("osm3d/vegetation", (0.35, 0.60, 0.25),
                 specular=(0.05, 0.05, 0.05),
                 depth_bias=(2.0, 1.0))


def farmland() -> str:
    if _pack_available("soil"):
        return "osm3d/farmland_pbr"
    return _make("osm3d/farmland", (0.70, 0.58, 0.35),
                 depth_bias=(2.0, 1.0))


def sand() -> str:
    if _pack_available("sand"):
        return "osm3d/sand_pbr"
    return _make("osm3d/sand", (0.86, 0.80, 0.55),
                 depth_bias=(2.0, 1.0))


def rock() -> str:
    if _pack_available("rock"):
        return "osm3d/rock_pbr"
    return _make("osm3d/rock", (0.55, 0.52, 0.48),
                 depth_bias=(2.0, 1.0))


def residential() -> str:
    if _pack_available("paved"):
        return "osm3d/residential_pbr"
    return _make("osm3d/residential", (0.75, 0.68, 0.55),
                 depth_bias=(1.5, 1.0))


def commercial() -> str:
    if _pack_available("paved"):
        return "osm3d/commercial_pbr"
    return _make("osm3d/commercial", (0.78, 0.62, 0.45),
                 depth_bias=(1.5, 1.0))


def industrial() -> str:
    if _pack_available("paved"):
        return "osm3d/industrial_pbr"
    return _make("osm3d/industrial", (0.50, 0.50, 0.50),
                 depth_bias=(1.5, 1.0))


def paved_square() -> str:
    """City squares, pedestrian plazas, marketplaces — cobbled/flagged pave."""
    if _pack_available("paved"):
        return "osm3d/paved_square_pbr"
    return _make("osm3d/paved_square", (0.55, 0.52, 0.48),
                 depth_bias=(1.5, 1.0))


def trees() -> str:
    return _make("osm3d/trees", (0.25, 0.50, 0.18),
                 specular=(0.05, 0.05, 0.05))


def furniture_metal() -> str:
    """Dark brushed metal for lamp posts."""
    return _make("osm3d/furniture_metal", (0.18, 0.18, 0.20),
                 specular=(0.25, 0.25, 0.28))


def furniture_wood() -> str:
    """Warm wood for benches."""
    return _make("osm3d/furniture_wood", (0.42, 0.27, 0.15),
                 specular=(0.05, 0.05, 0.05))


def building_trim() -> str:
    """Neutral warm-stone trim band rendered below the roof of every building."""
    return _make("osm3d/building_trim", (0.82, 0.78, 0.70),
                 specular=(0.06, 0.06, 0.06))
