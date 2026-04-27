from __future__ import annotations

from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.geometry.builder import GeomBuilder
from procbuilding.params import Color

_SLAB_T = 0.15   # slab thickness (metres)
_T = 1.5         # tile size in metres — matches white_plaster_02 real-world scale (~1.5 m)


def _box_uv(a: float, b: float) -> list[tuple[float, float]]:
    """UV for a rectangular face with dimensions a × b (in metres)."""
    return [(0.0, 0.0), (a / _T, 0.0), (a / _T, b / _T), (0.0, b / _T)]


class BalconyComponent(BuildingComponent):
    """
    Balcony slab with three-sided solid railing.

    Geometry:
    - Slab : horizontal platform (5 faces: top, underside, front edge, two side
      edges).  Thickness is _SLAB_T, so the slab bottom is at z_floor - _SLAB_T.
    - Railing : three double-sided thin panels (front + left + right), spanning
      from z_floor to z_floor + railing_height.  Double-sided so the inner face
      is visible from above.
    - Back face is open (abutting the building wall) so the door is unobstructed.

    Args:
        cx, cy         : world-space centre of the balcony attachment on the wall.
        z_floor        : world Z of the balcony floor (= floor_i * floor_height).
        width          : span of the balcony along the wall face.
        protrusion     : how far the balcony sticks out from the wall.
        railing_height : height of the railing above z_floor.
        face           : building wall the balcony is attached to.
        slab_color, railing_color : RGBA vertex colours.
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        z_floor: float,
        width: float,
        protrusion: float,
        railing_height: float,
        face: str,
        slab_color: Color,
        railing_color: Color,
    ) -> None:
        self._cx = cx
        self._cy = cy
        self._z0 = z_floor
        self._w = width
        self._p = protrusion
        self._rh = railing_height
        self._face = face
        self._sc = slab_color
        self._rc = railing_color

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> NodePath:
        b = GeomBuilder("balcony")
        f = self._face
        cx, cy = self._cx, self._cy
        z0 = self._z0
        hw = self._w / 2
        p = self._p
        rh = self._rh
        sc, rc = self._sc, self._rc
        zsb = z0 - _SLAB_T       # slab bottom
        zrt = z0 + rh             # railing top

        if f == "south":
            xL, xR = cx - hw, cx + hw
            yb, yf = cy, cy - p
            self._south_geom(b, xL, xR, yb, yf, zsb, z0, zrt, sc, rc,
                             self._w, p, rh)
        elif f == "north":
            xL, xR = cx - hw, cx + hw
            yb, yf = cy, cy + p
            self._north_geom(b, xL, xR, yb, yf, zsb, z0, zrt, sc, rc,
                             self._w, p, rh)
        elif f == "east":
            yS, yN = cy - hw, cy + hw
            xb, xf = cx, cx + p
            self._east_geom(b, yS, yN, xb, xf, zsb, z0, zrt, sc, rc,
                            self._w, p, rh)
        else:  # west
            yS, yN = cy - hw, cy + hw
            xb, xf = cx, cx - p
            self._west_geom(b, yS, yN, xb, xf, zsb, z0, zrt, sc, rc,
                            self._w, p, rh)

        return NodePath(b.build())

    # ------------------------------------------------------------------
    # Per-orientation helpers  (all normals verified by cross product)
    # ------------------------------------------------------------------

    @staticmethod
    def _south_geom(b, xL, xR, yb, yf, zsb, z0, zrt, sc, rc, w, p, rh):
        # --- Slab ---
        b.add_quad([LPoint3f(xL, yf, z0), LPoint3f(xR, yf, z0),
                    LPoint3f(xR, yb, z0), LPoint3f(xL, yb, z0)],
                   LVector3f(0, 0, 1), sc, uvs=_box_uv(w, p))
        b.add_quad([LPoint3f(xR, yf, zsb), LPoint3f(xL, yf, zsb),
                    LPoint3f(xL, yb, zsb), LPoint3f(xR, yb, zsb)],
                   LVector3f(0, 0, -1), sc, uvs=_box_uv(w, p))
        b.add_quad([LPoint3f(xL, yf, zsb), LPoint3f(xR, yf, zsb),
                    LPoint3f(xR, yf, z0), LPoint3f(xL, yf, z0)],
                   LVector3f(0, -1, 0), sc, uvs=_box_uv(w, _SLAB_T))
        b.add_quad([LPoint3f(xL, yb, zsb), LPoint3f(xL, yf, zsb),
                    LPoint3f(xL, yf, z0), LPoint3f(xL, yb, z0)],
                   LVector3f(-1, 0, 0), sc, uvs=_box_uv(p, _SLAB_T))
        b.add_quad([LPoint3f(xR, yf, zsb), LPoint3f(xR, yb, zsb),
                    LPoint3f(xR, yb, z0), LPoint3f(xR, yf, z0)],
                   LVector3f(1, 0, 0), sc, uvs=_box_uv(p, _SLAB_T))
        # --- Railing ---
        b.add_quad([LPoint3f(xL, yf, z0), LPoint3f(xR, yf, z0),
                    LPoint3f(xR, yf, zrt), LPoint3f(xL, yf, zrt)],
                   LVector3f(0, -1, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xR, yf, z0), LPoint3f(xL, yf, z0),
                    LPoint3f(xL, yf, zrt), LPoint3f(xR, yf, zrt)],
                   LVector3f(0, 1, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xL, yb, z0), LPoint3f(xL, yf, z0),
                    LPoint3f(xL, yf, zrt), LPoint3f(xL, yb, zrt)],
                   LVector3f(-1, 0, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xL, yf, z0), LPoint3f(xL, yb, z0),
                    LPoint3f(xL, yb, zrt), LPoint3f(xL, yf, zrt)],
                   LVector3f(1, 0, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xR, yf, z0), LPoint3f(xR, yb, z0),
                    LPoint3f(xR, yb, zrt), LPoint3f(xR, yf, zrt)],
                   LVector3f(1, 0, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xR, yb, z0), LPoint3f(xR, yf, z0),
                    LPoint3f(xR, yf, zrt), LPoint3f(xR, yb, zrt)],
                   LVector3f(-1, 0, 0), rc, uvs=_box_uv(p, rh))

    @staticmethod
    def _north_geom(b, xL, xR, yb, yf, zsb, z0, zrt, sc, rc, w, p, rh):
        b.add_quad([LPoint3f(xL, yb, z0), LPoint3f(xR, yb, z0),
                    LPoint3f(xR, yf, z0), LPoint3f(xL, yf, z0)],
                   LVector3f(0, 0, 1), sc, uvs=_box_uv(w, p))
        b.add_quad([LPoint3f(xR, yb, zsb), LPoint3f(xL, yb, zsb),
                    LPoint3f(xL, yf, zsb), LPoint3f(xR, yf, zsb)],
                   LVector3f(0, 0, -1), sc, uvs=_box_uv(w, p))
        b.add_quad([LPoint3f(xR, yf, zsb), LPoint3f(xL, yf, zsb),
                    LPoint3f(xL, yf, z0), LPoint3f(xR, yf, z0)],
                   LVector3f(0, 1, 0), sc, uvs=_box_uv(w, _SLAB_T))
        b.add_quad([LPoint3f(xL, yf, zsb), LPoint3f(xL, yb, zsb),
                    LPoint3f(xL, yb, z0), LPoint3f(xL, yf, z0)],
                   LVector3f(-1, 0, 0), sc, uvs=_box_uv(p, _SLAB_T))
        b.add_quad([LPoint3f(xR, yb, zsb), LPoint3f(xR, yf, zsb),
                    LPoint3f(xR, yf, z0), LPoint3f(xR, yb, z0)],
                   LVector3f(1, 0, 0), sc, uvs=_box_uv(p, _SLAB_T))
        # Railing
        b.add_quad([LPoint3f(xR, yf, z0), LPoint3f(xL, yf, z0),
                    LPoint3f(xL, yf, zrt), LPoint3f(xR, yf, zrt)],
                   LVector3f(0, 1, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xL, yf, z0), LPoint3f(xR, yf, z0),
                    LPoint3f(xR, yf, zrt), LPoint3f(xL, yf, zrt)],
                   LVector3f(0, -1, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xL, yf, z0), LPoint3f(xL, yb, z0),
                    LPoint3f(xL, yb, zrt), LPoint3f(xL, yf, zrt)],
                   LVector3f(-1, 0, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xL, yb, z0), LPoint3f(xL, yf, z0),
                    LPoint3f(xL, yf, zrt), LPoint3f(xL, yb, zrt)],
                   LVector3f(1, 0, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xR, yb, z0), LPoint3f(xR, yf, z0),
                    LPoint3f(xR, yf, zrt), LPoint3f(xR, yb, zrt)],
                   LVector3f(1, 0, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xR, yf, z0), LPoint3f(xR, yb, z0),
                    LPoint3f(xR, yb, zrt), LPoint3f(xR, yf, zrt)],
                   LVector3f(-1, 0, 0), rc, uvs=_box_uv(p, rh))

    @staticmethod
    def _east_geom(b, yS, yN, xb, xf, zsb, z0, zrt, sc, rc, w, p, rh):
        b.add_quad([LPoint3f(xb, yS, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yN, z0), LPoint3f(xb, yN, z0)],
                   LVector3f(0, 0, 1), sc, uvs=_box_uv(p, w))
        b.add_quad([LPoint3f(xf, yS, zsb), LPoint3f(xb, yS, zsb),
                    LPoint3f(xb, yN, zsb), LPoint3f(xf, yN, zsb)],
                   LVector3f(0, 0, -1), sc, uvs=_box_uv(p, w))
        b.add_quad([LPoint3f(xf, yS, zsb), LPoint3f(xf, yN, zsb),
                    LPoint3f(xf, yN, z0), LPoint3f(xf, yS, z0)],
                   LVector3f(1, 0, 0), sc, uvs=_box_uv(w, _SLAB_T))
        b.add_quad([LPoint3f(xb, yS, zsb), LPoint3f(xf, yS, zsb),
                    LPoint3f(xf, yS, z0), LPoint3f(xb, yS, z0)],
                   LVector3f(0, -1, 0), sc, uvs=_box_uv(p, _SLAB_T))
        b.add_quad([LPoint3f(xf, yN, zsb), LPoint3f(xb, yN, zsb),
                    LPoint3f(xb, yN, z0), LPoint3f(xf, yN, z0)],
                   LVector3f(0, 1, 0), sc, uvs=_box_uv(p, _SLAB_T))
        # Railing
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xf, yN, z0),
                    LPoint3f(xf, yN, zrt), LPoint3f(xf, yS, zrt)],
                   LVector3f(1, 0, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xf, yN, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yS, zrt), LPoint3f(xf, yN, zrt)],
                   LVector3f(-1, 0, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xb, yS, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yS, zrt), LPoint3f(xb, yS, zrt)],
                   LVector3f(0, -1, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xb, yS, z0),
                    LPoint3f(xb, yS, zrt), LPoint3f(xf, yS, zrt)],
                   LVector3f(0, 1, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xf, yN, z0), LPoint3f(xb, yN, z0),
                    LPoint3f(xb, yN, zrt), LPoint3f(xf, yN, zrt)],
                   LVector3f(0, 1, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xb, yN, z0), LPoint3f(xf, yN, z0),
                    LPoint3f(xf, yN, zrt), LPoint3f(xb, yN, zrt)],
                   LVector3f(0, -1, 0), rc, uvs=_box_uv(p, rh))

    @staticmethod
    def _west_geom(b, yS, yN, xb, xf, zsb, z0, zrt, sc, rc, w, p, rh):
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xb, yS, z0),
                    LPoint3f(xb, yN, z0), LPoint3f(xf, yN, z0)],
                   LVector3f(0, 0, 1), sc, uvs=_box_uv(p, w))
        b.add_quad([LPoint3f(xb, yS, zsb), LPoint3f(xf, yS, zsb),
                    LPoint3f(xf, yN, zsb), LPoint3f(xb, yN, zsb)],
                   LVector3f(0, 0, -1), sc, uvs=_box_uv(p, w))
        b.add_quad([LPoint3f(xf, yN, zsb), LPoint3f(xf, yS, zsb),
                    LPoint3f(xf, yS, z0), LPoint3f(xf, yN, z0)],
                   LVector3f(-1, 0, 0), sc, uvs=_box_uv(w, _SLAB_T))
        b.add_quad([LPoint3f(xf, yS, zsb), LPoint3f(xb, yS, zsb),
                    LPoint3f(xb, yS, z0), LPoint3f(xf, yS, z0)],
                   LVector3f(0, -1, 0), sc, uvs=_box_uv(p, _SLAB_T))
        b.add_quad([LPoint3f(xb, yN, zsb), LPoint3f(xf, yN, zsb),
                    LPoint3f(xf, yN, z0), LPoint3f(xb, yN, z0)],
                   LVector3f(0, 1, 0), sc, uvs=_box_uv(p, _SLAB_T))
        # Railing
        b.add_quad([LPoint3f(xf, yN, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yS, zrt), LPoint3f(xf, yN, zrt)],
                   LVector3f(-1, 0, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xf, yN, z0),
                    LPoint3f(xf, yN, zrt), LPoint3f(xf, yS, zrt)],
                   LVector3f(1, 0, 0), rc, uvs=_box_uv(w, rh))
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xb, yS, z0),
                    LPoint3f(xb, yS, zrt), LPoint3f(xf, yS, zrt)],
                   LVector3f(0, -1, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xb, yS, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yS, zrt), LPoint3f(xb, yS, zrt)],
                   LVector3f(0, 1, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xb, yN, z0), LPoint3f(xf, yN, z0),
                    LPoint3f(xf, yN, zrt), LPoint3f(xb, yN, zrt)],
                   LVector3f(0, 1, 0), rc, uvs=_box_uv(p, rh))
        b.add_quad([LPoint3f(xf, yN, z0), LPoint3f(xb, yN, z0),
                    LPoint3f(xb, yN, zrt), LPoint3f(xf, yN, zrt)],
                   LVector3f(0, -1, 0), rc, uvs=_box_uv(p, rh))
