"""PolygonSlab — a triangulated horizontal floor for an arbitrary 2D polygon."""
from __future__ import annotations

from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.geometry.builder import GeomBuilder
from procbuilding.geometry.polygon import Vert2D, ear_clip_triangulate
from procbuilding.params import Color

_UP = LVector3f(0, 0, 1)
_T = 1.0   # tile size in metres — matches concrete_floor_02 real-world scale (~1 m)


class PolygonSlab(BuildingComponent):
    """Horizontal slab whose footprint is an arbitrary CCW polygon.

    The polygon is triangulated via ear clipping and each triangle is added
    as a face with an upward normal.
    """

    def __init__(
        self,
        verts: list[Vert2D],
        z: float,
        color: Color,
        name: str = "polygon_slab",
    ) -> None:
        self._verts = verts
        self._z = z
        self._color = color
        self._name = name

    def build(self) -> NodePath:
        b = GeomBuilder(self._name)
        z = self._z
        for a, bv, c in ear_clip_triangulate(self._verts):
            b.add_triangle(
                [
                    LPoint3f(a[0],  a[1],  z),
                    LPoint3f(bv[0], bv[1], z),
                    LPoint3f(c[0],  c[1],  z),
                ],
                _UP,
                self._color,
                uvs=[(a[0]/_T, a[1]/_T), (bv[0]/_T, bv[1]/_T), (c[0]/_T, c[1]/_T)],
            )
        return NodePath(b.build())
