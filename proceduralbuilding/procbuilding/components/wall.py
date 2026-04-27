from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.geometry.builder import GeomBuilder, UV
from procbuilding.params import Color

_T = 1.5   # tile size in metres — matches white_plaster_02 real-world scale (~1.5 m)


@dataclass
class OpeningSpec:
    """Describes a window or door cut-out on a wall panel."""
    kind: Literal["window", "door"]
    center_x: float      # offset from wall left edge
    bottom_z: float      # offset from wall bottom
    width: float
    height: float
    recess: float        # how far inward the inner face is pushed
    color: Color


_NORMALS: dict[str, LVector3f] = {
    "south": LVector3f(0, -1, 0),
    "north": LVector3f(0,  1, 0),
    "west":  LVector3f(-1, 0, 0),
    "east":  LVector3f( 1, 0, 0),
}


class WallPanel(BuildingComponent):
    """
    One rectangular wall panel, optionally with recessed window/door openings.

    Openings are rendered as a recessed inner quad plus four reveal quads
    (left jamb, right jamb, head, sill). No actual mesh holes are cut.

    Args:
        origin:   bottom-left corner of the panel in world space
        width:    horizontal span of the panel
        height:   vertical span of the panel
        face:     cardinal direction this wall faces ("south","north","east","west")
        color:    RGBA vertex color for the wall surface
        openings: list of OpeningSpec to embed in the panel
    """

    def __init__(
        self,
        origin: LPoint3f,
        width: float,
        height: float,
        face: str,
        color: Color,
        openings: list[OpeningSpec] | None = None,
    ) -> None:
        self._origin = origin
        self._width = width
        self._height = height
        self._face = face
        self._color = color
        self._openings = openings or []
        self._normal = _NORMALS[face]

    # ------------------------------------------------------------------
    # Coordinate helpers — returns world-space points for this wall face
    # ------------------------------------------------------------------

    def _pt(self, u: float, v: float) -> LPoint3f:
        """Map (u=horizontal, v=vertical) local coords to world space.

        u increases in the direction that gives CCW winding when viewed from outside:
          south → +X,  north → -X (origin is at the far-right corner from outside)
          east  → +Y,  west  → -Y (origin is at the far-right corner from outside)
        """
        o = self._origin
        f = self._face
        if f == "south":
            return LPoint3f(o.x + u, o.y, o.z + v)
        elif f == "north":
            return LPoint3f(o.x - u, o.y, o.z + v)
        elif f == "east":
            return LPoint3f(o.x, o.y + u, o.z + v)
        else:  # west
            return LPoint3f(o.x, o.y - u, o.z + v)

    def _recess_pt(self, u: float, v: float, recess: float) -> LPoint3f:
        """Same as _pt but pushed inward by *recess* along the face normal."""
        p = self._pt(u, v)
        n = self._normal
        # inward is opposite to outward normal
        return LPoint3f(p.x - n.x * recess, p.y - n.y * recess, p.z - n.z * recess)

    @staticmethod
    def _uv(u: float, v: float) -> UV:
        return (u / _T, v / _T)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> NodePath:
        b = GeomBuilder(f"wall_{self._face}")
        n = self._normal
        child_builders: list[GeomBuilder] = []

        self._build_background(b)

        for op in self._openings:
            ox = op.center_x - op.width / 2
            oz = op.bottom_z
            ow, oh, r = op.width, op.height, op.recess
            oc = op.color
            wc = self._color  # wall color for reveals

            # Inner (recessed) face
            inner_verts = [
                self._recess_pt(ox,      oz,      r),
                self._recess_pt(ox + ow, oz,      r),
                self._recess_pt(ox + ow, oz + oh, r),
                self._recess_pt(ox,      oz + oh, r),
            ]
            inner_uvs = [self._uv(ox, oz), self._uv(ox+ow, oz),
                         self._uv(ox+ow, oz+oh), self._uv(ox, oz+oh)]

            if op.kind == "window":
                # Window glass goes in its own node so the viewer can apply glass material
                bg = GeomBuilder("glass")
                bg.add_quad(inner_verts, n, oc, uvs=inner_uvs)
                child_builders.append(bg)
            elif op.kind == "door":
                # Door panel goes in its own node so the viewer can apply wood material
                bd = GeomBuilder("door")
                bd.add_quad(inner_verts, n, oc, uvs=inner_uvs)
                child_builders.append(bd)
            else:
                b.add_quad(inner_verts, n, oc, uvs=inner_uvs)

            # Left jamb (facing right = +U direction)
            jamb_n_l = self._jamb_normal(right=False)
            b.add_quad(
                [
                    self._pt(ox,      oz),
                    self._recess_pt(ox, oz,      r),
                    self._recess_pt(ox, oz + oh, r),
                    self._pt(ox,      oz + oh),
                ],
                jamb_n_l, wc,
                uvs=[(0.0, oz/_T), (r/_T, oz/_T), (r/_T, (oz+oh)/_T), (0.0, (oz+oh)/_T)],
            )

            # Right jamb (facing left = -U direction)
            jamb_n_r = self._jamb_normal(right=True)
            b.add_quad(
                [
                    self._recess_pt(ox + ow, oz,      r),
                    self._pt(ox + ow,        oz),
                    self._pt(ox + ow,        oz + oh),
                    self._recess_pt(ox + ow, oz + oh, r),
                ],
                jamb_n_r, wc,
                uvs=[(r/_T, oz/_T), (0.0, oz/_T), (0.0, (oz+oh)/_T), (r/_T, (oz+oh)/_T)],
            )

            # Head (top reveal, facing downward)
            b.add_quad(
                [
                    self._recess_pt(ox,      oz + oh, r),
                    self._recess_pt(ox + ow, oz + oh, r),
                    self._pt(ox + ow,        oz + oh),
                    self._pt(ox,             oz + oh),
                ],
                LVector3f(0, 0, -1), wc,
                uvs=[self._uv(ox, 0), self._uv(ox+ow, 0),
                     self._uv(ox+ow, r), self._uv(ox, r)],
            )

            # Sill (bottom reveal, facing upward) — only for windows
            if op.kind == "window":
                b.add_quad(
                    [
                        self._pt(ox,             oz),
                        self._pt(ox + ow,        oz),
                        self._recess_pt(ox + ow, oz, r),
                        self._recess_pt(ox,      oz, r),
                    ],
                    LVector3f(0, 0, 1), wc,
                    uvs=[self._uv(ox, 0), self._uv(ox+ow, 0),
                         self._uv(ox+ow, r), self._uv(ox, r)],
                )

        wall_np = NodePath(b.build())
        for bg in child_builders:
            NodePath(bg.build()).reparentTo(wall_np)
        return wall_np

    def _build_background(self, b: GeomBuilder) -> None:
        """Tile the wall face with quads that leave gaps for every opening.

        Uses a horizontal sweep-line over z-breakpoints so mixed window/door
        heights are handled correctly without any special-casing.
        """
        w, h = self._width, self._height
        n, c = self._normal, self._color

        if not self._openings:
            b.add_quad(
                [self._pt(0, 0), self._pt(w, 0), self._pt(w, h), self._pt(0, h)],
                n, c,
                uvs=[self._uv(0, 0), self._uv(w, 0), self._uv(w, h), self._uv(0, h)],
            )
            return

        # (u_left, u_right, v_bottom, v_top) for each opening
        rects = [
            (
                op.center_x - op.width / 2,
                op.center_x + op.width / 2,
                op.bottom_z,
                op.bottom_z + op.height,
            )
            for op in self._openings
        ]

        # All horizontal cut lines
        z_cuts = sorted({0.0, h} | {v for r in rects for v in (r[2], r[3])})

        for z0, z1 in zip(z_cuts, z_cuts[1:]):
            if z0 >= z1:
                continue
            z_mid = (z0 + z1) * 0.5
            # Openings whose v-range straddles this band's midpoint
            blocked = sorted(
                (xl, xr) for xl, xr, vb, vt in rects if vb <= z_mid < vt
            )
            u = 0.0
            for xl, xr in blocked:
                if xl > u:
                    b.add_quad(
                        [self._pt(u, z0), self._pt(xl, z0),
                         self._pt(xl, z1), self._pt(u, z1)],
                        n, c,
                        uvs=[self._uv(u, z0), self._uv(xl, z0),
                             self._uv(xl, z1), self._uv(u, z1)],
                    )
                u = max(u, xr)
            if u < w:
                b.add_quad(
                    [self._pt(u, z0), self._pt(w, z0),
                     self._pt(w, z1), self._pt(u, z1)],
                    n, c,
                    uvs=[self._uv(u, z0), self._uv(w, z0),
                         self._uv(w, z1), self._uv(u, z1)],
                )

    def _jamb_normal(self, right: bool) -> LVector3f:
        """Normal for a jamb reveal face.

        The left jamb (right=False) faces in the +u direction (into the opening).
        The right jamb (right=True) faces in the -u direction.
        u-directions: south→+X, north→-X, east→+Y, west→-Y.
        """
        f = self._face
        if f == "south":
            return LVector3f(-1 if right else  1,  0, 0)
        elif f == "north":
            return LVector3f( 1 if right else -1,  0, 0)
        elif f == "east":
            return LVector3f(0, -1 if right else  1, 0)
        else:  # west
            return LVector3f(0,  1 if right else -1, 0)
