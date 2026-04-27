from __future__ import annotations

from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.geometry.builder import GeomBuilder
from procbuilding.params import Color

_T = 0.4   # tile size in metres — brick pattern scale (~0.4 m per 2 courses)


class ChimneyComponent(BuildingComponent):
    """
    Rectangular masonry chimney stack.

    Runs from z=0 (ground) to z_eave + height so it visually pierces
    the roof and protrudes above it.  All four vertical sides plus a
    top cap are rendered; the bottom is left open (sits on the slab).
    """

    def __init__(
        self,
        x_center: float,
        y_center: float,
        width: float,
        depth: float,
        z_eave: float,
        height: float,
        color: Color,
        cap_color: Color,
    ) -> None:
        self._xc = x_center
        self._yc = y_center
        self._w = width
        self._d = depth
        self._z1 = z_eave + height
        self._color = color
        self._cap_color = cap_color

    def build(self) -> NodePath:
        b = GeomBuilder("chimney")
        x0 = self._xc - self._w / 2
        x1 = self._xc + self._w / 2
        y0 = self._yc - self._d / 2
        y1 = self._yc + self._d / 2
        z0 = 0.0
        z1 = self._z1
        c = self._color
        cc = self._cap_color
        w, d, h = x1 - x0, y1 - y0, z1 - z0

        # South face  normal (0,-1,0)  — u along X, v along Z
        b.add_quad(
            [LPoint3f(x0, y0, z0), LPoint3f(x1, y0, z0),
             LPoint3f(x1, y0, z1), LPoint3f(x0, y0, z1)],
            LVector3f(0, -1, 0), c,
            uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, h/_T), (0.0, h/_T)],
        )
        # North face  normal (0,+1,0)
        b.add_quad(
            [LPoint3f(x1, y1, z0), LPoint3f(x0, y1, z0),
             LPoint3f(x0, y1, z1), LPoint3f(x1, y1, z1)],
            LVector3f(0, 1, 0), c,
            uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, h/_T), (0.0, h/_T)],
        )
        # West face  normal (-1,0,0)  — u along Y, v along Z
        b.add_quad(
            [LPoint3f(x0, y1, z0), LPoint3f(x0, y0, z0),
             LPoint3f(x0, y0, z1), LPoint3f(x0, y1, z1)],
            LVector3f(-1, 0, 0), c,
            uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)],
        )
        # East face  normal (+1,0,0)
        b.add_quad(
            [LPoint3f(x1, y0, z0), LPoint3f(x1, y1, z0),
             LPoint3f(x1, y1, z1), LPoint3f(x1, y0, z1)],
            LVector3f(1, 0, 0), c,
            uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)],
        )
        # Top cap  normal (0,0,+1)  — u along X, v along Y
        b.add_quad(
            [LPoint3f(x0, y0, z1), LPoint3f(x1, y0, z1),
             LPoint3f(x1, y1, z1), LPoint3f(x0, y1, z1)],
            LVector3f(0, 0, 1), cc,
            uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, d/_T), (0.0, d/_T)],
        )

        return NodePath(b.build())
