from __future__ import annotations

from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.geometry.builder import GeomBuilder
from procbuilding.params import Color

_T = 0.5   # tile size in metres — metal plate panels are small (~0.5 m)


class ACUnit(BuildingComponent):
    """
    Rectangular box representing an AC external unit mounted flush against a wall.

    The front face (grille side) faces outward from the building; the back face
    is omitted because it is hidden inside the wall.  Five faces total:
    front (grille_color) + two sides + top + bottom (all in color).

    Args:
        cx, cy : world-space centre of the unit's back face on the wall surface.
        z_bottom: world Z of the unit's bottom edge.
        width   : span of the unit along the wall face.
        depth   : how far the unit protrudes from the wall.
        height  : vertical span of the unit.
        face    : wall the unit is mounted on ("south","north","east","west").
        color       : body color.
        grille_color: front face color (typically a little darker).
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        z_bottom: float,
        width: float,
        depth: float,
        height: float,
        face: str,
        color: Color,
        grille_color: Color,
    ) -> None:
        self._cx = cx
        self._cy = cy
        self._z0 = z_bottom
        self._w = width
        self._d = depth
        self._h = height
        self._face = face
        self._color = color
        self._grille_color = grille_color

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> NodePath:
        b = GeomBuilder("ac_unit")
        f = self._face
        cx, cy = self._cx, self._cy
        z0, z1 = self._z0, self._z0 + self._h
        hw = self._w / 2          # half-width along wall face
        d = self._d
        c, gc = self._color, self._grille_color
        w, h = self._w, self._h

        if f == "south":
            # Wall at y=cy (=0).  Unit protrudes toward -Y.
            xL, xR = cx - hw, cx + hw
            y_back, y_front = cy, cy - d
            self._south_faces(b, xL, xR, y_back, y_front, z0, z1, c, gc, w, d, h)

        elif f == "north":
            # Wall at y=cy (=depth).  Unit protrudes toward +Y.
            xL, xR = cx - hw, cx + hw
            y_back, y_front = cy, cy + d
            self._north_faces(b, xL, xR, y_back, y_front, z0, z1, c, gc, w, d, h)

        elif f == "east":
            # Wall at x=cx (=width).  Unit protrudes toward +X.
            yS, yN = cy - hw, cy + hw
            x_back, x_front = cx, cx + d
            self._east_faces(b, yS, yN, x_back, x_front, z0, z1, c, gc, w, d, h)

        else:  # west
            # Wall at x=cx (=0).  Unit protrudes toward -X.
            yS, yN = cy - hw, cy + hw
            x_back, x_front = cx, cx - d
            self._west_faces(b, yS, yN, x_back, x_front, z0, z1, c, gc, w, d, h)

        return NodePath(b.build())

    # ------------------------------------------------------------------
    # Per-orientation helpers  (all normals verified by cross product)
    # ------------------------------------------------------------------

    @staticmethod
    def _south_faces(b, xL, xR, yb, yf, z0, z1, c, gc, w, d, h):
        # Front  (0,-1,0)
        b.add_quad([LPoint3f(xL, yf, z0), LPoint3f(xR, yf, z0),
                    LPoint3f(xR, yf, z1), LPoint3f(xL, yf, z1)],
                   LVector3f(0, -1, 0), gc,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, h/_T), (0.0, h/_T)])
        # West side  (-1,0,0)
        b.add_quad([LPoint3f(xL, yb, z0), LPoint3f(xL, yf, z0),
                    LPoint3f(xL, yf, z1), LPoint3f(xL, yb, z1)],
                   LVector3f(-1, 0, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # East side  (+1,0,0)
        b.add_quad([LPoint3f(xR, yf, z0), LPoint3f(xR, yb, z0),
                    LPoint3f(xR, yb, z1), LPoint3f(xR, yf, z1)],
                   LVector3f(1, 0, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # Top  (0,0,+1)
        b.add_quad([LPoint3f(xL, yf, z1), LPoint3f(xR, yf, z1),
                    LPoint3f(xR, yb, z1), LPoint3f(xL, yb, z1)],
                   LVector3f(0, 0, 1), c,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, d/_T), (0.0, d/_T)])
        # Bottom  (0,0,-1)
        b.add_quad([LPoint3f(xL, yb, z0), LPoint3f(xR, yb, z0),
                    LPoint3f(xR, yf, z0), LPoint3f(xL, yf, z0)],
                   LVector3f(0, 0, -1), c,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, d/_T), (0.0, d/_T)])

    @staticmethod
    def _north_faces(b, xL, xR, yb, yf, z0, z1, c, gc, w, d, h):
        # Front  (0,+1,0)
        b.add_quad([LPoint3f(xR, yf, z0), LPoint3f(xL, yf, z0),
                    LPoint3f(xL, yf, z1), LPoint3f(xR, yf, z1)],
                   LVector3f(0, 1, 0), gc,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, h/_T), (0.0, h/_T)])
        # West side  (-1,0,0)
        b.add_quad([LPoint3f(xL, yf, z0), LPoint3f(xL, yb, z0),
                    LPoint3f(xL, yb, z1), LPoint3f(xL, yf, z1)],
                   LVector3f(-1, 0, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # East side  (+1,0,0)
        b.add_quad([LPoint3f(xR, yb, z0), LPoint3f(xR, yf, z0),
                    LPoint3f(xR, yf, z1), LPoint3f(xR, yb, z1)],
                   LVector3f(1, 0, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # Top  (0,0,+1)
        b.add_quad([LPoint3f(xL, yb, z1), LPoint3f(xR, yb, z1),
                    LPoint3f(xR, yf, z1), LPoint3f(xL, yf, z1)],
                   LVector3f(0, 0, 1), c,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, d/_T), (0.0, d/_T)])
        # Bottom  (0,0,-1)
        b.add_quad([LPoint3f(xR, yb, z0), LPoint3f(xL, yb, z0),
                    LPoint3f(xL, yf, z0), LPoint3f(xR, yf, z0)],
                   LVector3f(0, 0, -1), c,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, d/_T), (0.0, d/_T)])

    @staticmethod
    def _east_faces(b, yS, yN, xb, xf, z0, z1, c, gc, w, d, h):
        # Front  (+1,0,0)
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xf, yN, z0),
                    LPoint3f(xf, yN, z1), LPoint3f(xf, yS, z1)],
                   LVector3f(1, 0, 0), gc,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, h/_T), (0.0, h/_T)])
        # South side  (0,-1,0)
        b.add_quad([LPoint3f(xb, yS, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yS, z1), LPoint3f(xb, yS, z1)],
                   LVector3f(0, -1, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # North side  (0,+1,0)
        b.add_quad([LPoint3f(xf, yN, z0), LPoint3f(xb, yN, z0),
                    LPoint3f(xb, yN, z1), LPoint3f(xf, yN, z1)],
                   LVector3f(0, 1, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # Top  (0,0,+1)
        b.add_quad([LPoint3f(xb, yS, z1), LPoint3f(xf, yS, z1),
                    LPoint3f(xf, yN, z1), LPoint3f(xb, yN, z1)],
                   LVector3f(0, 0, 1), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, w/_T), (0.0, w/_T)])
        # Bottom  (0,0,-1)
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xb, yS, z0),
                    LPoint3f(xb, yN, z0), LPoint3f(xf, yN, z0)],
                   LVector3f(0, 0, -1), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, w/_T), (0.0, w/_T)])

    @staticmethod
    def _west_faces(b, yS, yN, xb, xf, z0, z1, c, gc, w, d, h):
        # Front  (-1,0,0)
        b.add_quad([LPoint3f(xf, yN, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yS, z1), LPoint3f(xf, yN, z1)],
                   LVector3f(-1, 0, 0), gc,
                   uvs=[(0.0, 0.0), (w/_T, 0.0), (w/_T, h/_T), (0.0, h/_T)])
        # South side  (0,-1,0)
        b.add_quad([LPoint3f(xf, yS, z0), LPoint3f(xb, yS, z0),
                    LPoint3f(xb, yS, z1), LPoint3f(xf, yS, z1)],
                   LVector3f(0, -1, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # North side  (0,+1,0)
        b.add_quad([LPoint3f(xb, yN, z0), LPoint3f(xf, yN, z0),
                    LPoint3f(xf, yN, z1), LPoint3f(xb, yN, z1)],
                   LVector3f(0, 1, 0), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, h/_T), (0.0, h/_T)])
        # Top  (0,0,+1)
        b.add_quad([LPoint3f(xf, yS, z1), LPoint3f(xb, yS, z1),
                    LPoint3f(xb, yN, z1), LPoint3f(xf, yN, z1)],
                   LVector3f(0, 0, 1), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, w/_T), (0.0, w/_T)])
        # Bottom  (0,0,-1)
        b.add_quad([LPoint3f(xb, yS, z0), LPoint3f(xf, yS, z0),
                    LPoint3f(xf, yN, z0), LPoint3f(xb, yN, z0)],
                   LVector3f(0, 0, -1), c,
                   uvs=[(0.0, 0.0), (d/_T, 0.0), (d/_T, w/_T), (0.0, w/_T)])
