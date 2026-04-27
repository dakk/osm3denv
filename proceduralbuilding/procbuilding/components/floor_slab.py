from __future__ import annotations
from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.geometry.builder import GeomBuilder
from procbuilding.params import Color

_T = 1.0   # tile size in metres — matches concrete_floor_02 real-world scale (~1 m)


class FloorSlab(BuildingComponent):
    """Horizontal quad representing a floor or ceiling surface."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        width: float,
        depth: float,
        color: Color,
        name: str = "floor_slab",
    ) -> None:
        self._x = x
        self._y = y
        self._z = z
        self._width = width
        self._depth = depth
        self._color = color
        self._name = name

    def build(self) -> NodePath:
        b = GeomBuilder(self._name)
        x, y, z = self._x, self._y, self._z
        w, d = self._width, self._depth
        b.add_quad(
            [
                LPoint3f(x,     y,     z),
                LPoint3f(x + w, y,     z),
                LPoint3f(x + w, y + d, z),
                LPoint3f(x,     y + d, z),
            ],
            LVector3f(0, 0, 1),
            self._color,
            uvs=[(x/_T, y/_T), ((x+w)/_T, y/_T), ((x+w)/_T, (y+d)/_T), (x/_T, (y+d)/_T)],
        )
        return NodePath(b.build())
