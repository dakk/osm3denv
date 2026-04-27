from __future__ import annotations
import math
from abc import ABC, abstractmethod

from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.geometry.builder import GeomBuilder, face_normal
from procbuilding.params import Color, RoofType

_T = 1.2   # tile size in metres — matches clay_roof_tiles_02 real-world scale (~1.2 m)


class _RoofStrategy(ABC):
    @abstractmethod
    def build(self, b: GeomBuilder, x0: float, y0: float, x1: float, y1: float,
              z_eave: float, pitch: float, overhang: float, color: Color) -> None:
        """Add all roof geometry to *b*."""


class _FlatRoof(_RoofStrategy):
    def build(self, b, x0, y0, x1, y1, z_eave, pitch, overhang, color):
        ox, oy = overhang, overhang
        xL = x0 - ox
        xR = x1 + ox
        yS = y0 - oy
        yN = y1 + oy
        b.add_quad(
            [
                LPoint3f(xL, yS, z_eave),
                LPoint3f(xR, yS, z_eave),
                LPoint3f(xR, yN, z_eave),
                LPoint3f(xL, yN, z_eave),
            ],
            LVector3f(0, 0, 1),
            color,
            uvs=[(xL/_T, yS/_T), (xR/_T, yS/_T), (xR/_T, yN/_T), (xL/_T, yN/_T)],
        )


class _GableRoof(_RoofStrategy):
    """
    Ridge runs along X. Two slope panels + two triangular gable ends.

    Cross-section (Y-Z plane):
        eave_south --- apex --- eave_north
    apex_z = z_eave + pitch * (depth / 2)
    apex_y = (y0 + y1) / 2
    """

    def build(self, b, x0, y0, x1, y1, z_eave, pitch, overhang, color):
        ox, oy = overhang, overhang
        xL = x0 - ox
        xR = x1 + ox
        y_south = y0 - oy
        y_north = y1 + oy
        y_apex = (y0 + y1) / 2
        z_apex = z_eave + pitch * ((y1 - y0) / 2)

        # Arc length along south slope (hypotenuse of pitch triangle)
        dy_s = y_apex - y_south
        dz_s = z_apex - z_eave
        slope_s = math.sqrt(dy_s * dy_s + dz_s * dz_s)

        # South slope (facing south+up)
        p0 = LPoint3f(xL, y_south, z_eave)
        p1 = LPoint3f(xR, y_south, z_eave)
        p2 = LPoint3f(xR, y_apex,  z_apex)
        p3 = LPoint3f(xL, y_apex,  z_apex)
        n = face_normal(p0, p1, p2)
        b.add_quad([p0, p1, p2, p3], n, color,
                   uvs=[(xL/_T, 0.0), (xR/_T, 0.0),
                        (xR/_T, slope_s/_T), (xL/_T, slope_s/_T)])

        # North slope (facing north+up) — V=0 at ridge (p0/p1), V=slope at eave
        p0 = LPoint3f(xL, y_apex,  z_apex)
        p1 = LPoint3f(xR, y_apex,  z_apex)
        p2 = LPoint3f(xR, y_north, z_eave)
        p3 = LPoint3f(xL, y_north, z_eave)
        n = face_normal(p0, p1, p2)
        b.add_quad([p0, p1, p2, p3], n, color,
                   uvs=[(xL/_T, slope_s/_T), (xR/_T, slope_s/_T),
                        (xR/_T, 0.0), (xL/_T, 0.0)])

        # West gable end (facing -X): CCW from west = [south, apex, north]
        g0 = LPoint3f(xL, y_south, z_eave)
        g1 = LPoint3f(xL, y_apex,  z_apex)
        g2 = LPoint3f(xL, y_north, z_eave)
        n = face_normal(g0, g1, g2)
        b.add_triangle([g0, g1, g2], n, color,
                       uvs=[(y_south/_T, z_eave/_T),
                            (y_apex/_T,  z_apex/_T),
                            (y_north/_T, z_eave/_T)])

        # East gable end (facing +X): CCW from east = [north, apex, south]
        g0 = LPoint3f(xR, y_north, z_eave)
        g1 = LPoint3f(xR, y_apex,  z_apex)
        g2 = LPoint3f(xR, y_south, z_eave)
        n = face_normal(g0, g1, g2)
        b.add_triangle([g0, g1, g2], n, color,
                       uvs=[(y_north/_T, z_eave/_T),
                            (y_apex/_T,  z_apex/_T),
                            (y_south/_T, z_eave/_T)])


class _HipRoof(_RoofStrategy):
    """
    Four-sided roof. Ridge is shorter than the building length.
    Ridge length = max(0, width - depth) along X.
    Two trapezoid slopes (long sides) + two triangular hip ends (short sides).
    """

    def build(self, b, x0, y0, x1, y1, z_eave, pitch, overhang, color):
        ox, oy = overhang, overhang
        xL = x0 - ox
        xR = x1 + ox
        y_south = y0 - oy
        y_north = y1 + oy
        bw = xR - xL           # full width with overhang
        bd = y_north - y_south  # full depth with overhang
        y_apex = (y_south + y_north) / 2
        z_apex = z_eave + pitch * (bd / 2)

        dy_s = y_apex - y_south
        dz_s = z_apex - z_eave
        slope_s = math.sqrt(dy_s * dy_s + dz_s * dz_s)

        # Ridge endpoints: if width > depth, ridge has non-zero length
        half_ridge = max(0.0, (bw - bd) / 2)
        xRL = (xL + xR) / 2 - half_ridge
        xRR = (xL + xR) / 2 + half_ridge

        if half_ridge > 0:
            # South trapezoid slope
            p0 = LPoint3f(xL,  y_south, z_eave)
            p1 = LPoint3f(xR,  y_south, z_eave)
            p2 = LPoint3f(xRR, y_apex,  z_apex)
            p3 = LPoint3f(xRL, y_apex,  z_apex)
            n = face_normal(p0, p1, p2)
            b.add_quad([p0, p1, p2, p3], n, color,
                       uvs=[(xL/_T, 0.0), (xR/_T, 0.0),
                            (xRR/_T, slope_s/_T), (xRL/_T, slope_s/_T)])

            # North trapezoid slope — V=slope at ridge, V=0 at eave
            p0 = LPoint3f(xRL, y_apex,  z_apex)
            p1 = LPoint3f(xRR, y_apex,  z_apex)
            p2 = LPoint3f(xR,  y_north, z_eave)
            p3 = LPoint3f(xL,  y_north, z_eave)
            n = face_normal(p0, p1, p2)
            b.add_quad([p0, p1, p2, p3], n, color,
                       uvs=[(xRL/_T, slope_s/_T), (xRR/_T, slope_s/_T),
                            (xR/_T, 0.0), (xL/_T, 0.0)])
        else:
            # Square/deep building — south and north become triangles
            xM = (xL + xR) / 2
            p0 = LPoint3f(xL, y_south, z_eave)
            p1 = LPoint3f(xR, y_south, z_eave)
            p2 = LPoint3f(xM, y_apex, z_apex)
            n = face_normal(p0, p1, p2)
            b.add_triangle([p0, p1, p2], n, color,
                           uvs=[(xL/_T, 0.0), (xR/_T, 0.0), (xM/_T, slope_s/_T)])

            # North triangle — V=0 at eave, V=slope at apex
            p0 = LPoint3f(xR, y_north, z_eave)
            p1 = LPoint3f(xL, y_north, z_eave)
            p2 = LPoint3f(xM, y_apex, z_apex)
            n = face_normal(p0, p1, p2)
            b.add_triangle([p0, p1, p2], n, color,
                           uvs=[(xR/_T, 0.0), (xL/_T, 0.0), (xM/_T, slope_s/_T)])

        # West hip triangle
        p0 = LPoint3f(xL, y_south, z_eave)
        p1 = LPoint3f(xRL, y_apex, z_apex)
        p2 = LPoint3f(xL, y_north, z_eave)
        n = face_normal(p0, p1, p2)
        b.add_triangle([p0, p1, p2], n, color,
                       uvs=[(y_south/_T, 0.0), (y_apex/_T, slope_s/_T), (y_north/_T, 0.0)])

        # East hip triangle
        p0 = LPoint3f(xR, y_north, z_eave)
        p1 = LPoint3f(xRR, y_apex, z_apex)
        p2 = LPoint3f(xR, y_south, z_eave)
        n = face_normal(p0, p1, p2)
        b.add_triangle([p0, p1, p2], n, color,
                       uvs=[(y_north/_T, 0.0), (y_apex/_T, slope_s/_T), (y_south/_T, 0.0)])


_STRATEGIES: dict[RoofType, type[_RoofStrategy]] = {
    RoofType.FLAT:  _FlatRoof,
    RoofType.GABLE: _GableRoof,
    RoofType.HIP:   _HipRoof,
}


class RoofComponent(BuildingComponent):
    """Builds a roof using the appropriate strategy for the requested RoofType."""

    def __init__(
        self,
        roof_type: RoofType,
        x0: float, y0: float,
        x1: float, y1: float,
        z_eave: float,
        pitch: float,
        overhang: float,
        color: Color,
    ) -> None:
        self._strategy = _STRATEGIES[roof_type]()
        self._args = (x0, y0, x1, y1, z_eave, pitch, overhang, color)

    def build(self) -> NodePath:
        b = GeomBuilder("roof")
        self._strategy.build(b, *self._args)
        return NodePath(b.build())
