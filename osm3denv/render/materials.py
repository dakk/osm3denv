"""Programmatic materials for each scene layer."""
from __future__ import annotations

import Ogre


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
    return _make("osm3d/terrain", (0.45, 0.55, 0.35))


def buildings() -> str:
    return _make("osm3d/buildings", (0.80, 0.75, 0.65),
                 specular=(0.10, 0.10, 0.10))


def roads() -> str:
    return _make("osm3d/roads", (0.22, 0.22, 0.22),
                 depth_bias=(10.0, 5.0))


def water() -> str:
    return _make("osm3d/water", (0.20, 0.35, 0.55), alpha=0.85,
                 depth_bias=(1.0, 1.0))
