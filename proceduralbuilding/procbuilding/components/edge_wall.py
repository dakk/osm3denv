"""EdgeWallPanel — a wall panel along an arbitrary 2D edge direction."""
from __future__ import annotations

import math

from panda3d.core import LPoint3f, LVector3f, NodePath

from procbuilding.components.base import BuildingComponent
from procbuilding.components.wall import OpeningSpec
from procbuilding.geometry.builder import GeomBuilder, UV
from procbuilding.params import Color

_T = 1.5   # tile size in metres — matches white_plaster_02 real-world scale (~1.5 m)


class EdgeWallPanel(BuildingComponent):
    """
    One wall panel aligned to an arbitrary 2D edge (p0 → p1).

    The outward normal is computed as the CCW-polygon outward direction:
        n = (edge_dir.y, -edge_dir.x)   (rotate edge direction by −90°).

    Openings use the same OpeningSpec as WallPanel: center_x is measured
    along the edge from p0, bottom_z from z_bottom.

    Args:
        p0, p1:  2D (x, y) world-space endpoints of the edge
        z_bottom: world-space Z of the wall bottom
        height:  wall height
        color:   RGBA vertex color for the wall surface
        openings: list of OpeningSpec to embed (center_x = offset from p0 along edge)
    """

    def __init__(
        self,
        p0: tuple[float, float],
        p1: tuple[float, float],
        z_bottom: float,
        height: float,
        color: Color,
        openings: list[OpeningSpec] | None = None,
    ) -> None:
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        self._length = math.sqrt(dx * dx + dy * dy)
        if self._length < 1e-9:
            raise ValueError("EdgeWallPanel: zero-length edge")
        self._p0 = p0
        self._ed = (dx / self._length, dy / self._length)   # unit edge direction
        self._n2 = (self._ed[1], -self._ed[0])              # 2D outward normal (CCW polygon)
        self._z0 = z_bottom
        self._h = height
        self._color = color
        self._openings = openings or []
        self._normal = LVector3f(self._n2[0], self._n2[1], 0.0)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _pt(self, u: float, v: float) -> LPoint3f:
        """Map (u=along-edge, v=height) to world space."""
        return LPoint3f(
            self._p0[0] + u * self._ed[0],
            self._p0[1] + u * self._ed[1],
            self._z0 + v,
        )

    def _recess_pt(self, u: float, v: float, recess: float) -> LPoint3f:
        """Same as _pt but pushed inward by *recess* (opposite to outward normal)."""
        p = self._pt(u, v)
        return LPoint3f(
            p.x - self._n2[0] * recess,
            p.y - self._n2[1] * recess,
            p.z,
        )

    @staticmethod
    def _uv(u: float, v: float) -> UV:
        return (u / _T, v / _T)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> NodePath:
        b = GeomBuilder("edge_wall")
        self._build_background(b)
        n = self._normal
        child_builders: list[GeomBuilder] = []

        for op in self._openings:
            ox = op.center_x - op.width / 2
            oz = op.bottom_z
            ow, oh, r = op.width, op.height, op.recess
            oc = op.color
            wc = self._color

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
                bg = GeomBuilder("glass")
                bg.add_quad(inner_verts, n, oc, uvs=inner_uvs)
                child_builders.append(bg)
            elif op.kind == "door":
                bd = GeomBuilder("door")
                bd.add_quad(inner_verts, n, oc, uvs=inner_uvs)
                child_builders.append(bd)
            else:
                b.add_quad(inner_verts, n, oc, uvs=inner_uvs)
            # Left jamb — faces +edge_dir (inward from outside-left perspective)
            jamb_n_l = LVector3f(self._ed[0], self._ed[1], 0.0)
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
            # Right jamb — faces -edge_dir
            jamb_n_r = LVector3f(-self._ed[0], -self._ed[1], 0.0)
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
            # Sill (bottom reveal, facing upward) — windows only
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
        """Tile the wall face with quads that leave gaps for every opening."""
        w, h = self._length, self._h
        n, c = self._normal, self._color

        if not self._openings:
            b.add_quad(
                [self._pt(0, 0), self._pt(w, 0), self._pt(w, h), self._pt(0, h)],
                n, c,
                uvs=[self._uv(0, 0), self._uv(w, 0), self._uv(w, h), self._uv(0, h)],
            )
            return

        rects = [
            (
                op.center_x - op.width / 2,
                op.center_x + op.width / 2,
                op.bottom_z,
                op.bottom_z + op.height,
            )
            for op in self._openings
        ]

        z_cuts = sorted({0.0, h} | {v for r in rects for v in (r[2], r[3])})

        for z0, z1 in zip(z_cuts, z_cuts[1:]):
            if z0 >= z1:
                continue
            z_mid = (z0 + z1) * 0.5
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
