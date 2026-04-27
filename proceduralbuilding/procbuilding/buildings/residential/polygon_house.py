"""PolygonHouse and LShapedHouse — buildings with non-rectangular footprints."""
from __future__ import annotations

import math

from panda3d.core import NodePath

from procbuilding.buildings.residential.base_house import (
    BaseHouse,
    add_ac_units,
    add_balcony,
    build_polygon_floors,
)
from procbuilding.buildings.residential.component_params import ACUnitParams, BalconyParams
from procbuilding.buildings.residential.polygon_params import (
    LShapedHouseParams,
    PolygonHouseParams,
)
from procbuilding.components.polygon_slab import PolygonSlab
from procbuilding.components.wall import OpeningSpec
from procbuilding.geometry.polygon import l_shape_verts
from procbuilding.registry import register_building

# Edge index → outer face name for the standard 6-vertex L-shape polygon
_L_EDGE_FACE: dict[int, str] = {0: "south", 1: "east", 4: "north", 5: "west"}


@register_building("polygon_house")
class PolygonHouse(BaseHouse):
    """
    Procedural building with an arbitrary polygon footprint.

    Every polygon edge becomes a wall panel; the floor slab is triangulated
    from the footprint.  Roof is always flat.  Chimney uses bounding-box coords.
    """

    name = "PolygonHouse"

    def __init__(self, params: PolygonHouseParams | None = None) -> None:
        if params is None:
            params = PolygonHouseParams(verts=[(0, 0), (10, 0), (10, 8), (0, 8)])
        self._p = params

    # ------------------------------------------------------------------
    # BaseHouse abstract implementation
    # ------------------------------------------------------------------

    def _chimney_xy(self) -> tuple[float, float]:
        ch = self._p.chimney
        verts = self._p.verts
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return (
            min(xs) + ch.pos_x * (max(xs) - min(xs)),
            min(ys) + ch.pos_y * (max(ys) - min(ys)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, parent: NodePath | None = None) -> NodePath:
        p = self._p
        root = NodePath("polygon_house")
        if parent is not None:
            root.reparentTo(parent)

        build_polygon_floors(
            root,
            verts=p.verts,
            num_floors=p.num_floors,
            floor_height=p.floor_height,
            floor_color=p.floor_color,
            wall_color=p.wall_color,
            openings_fn=self._openings_for,
        )

        z_eave = p.num_floors * p.floor_height
        PolygonSlab(p.verts, z_eave, p.roof_color, "roof").build().reparentTo(root)

        self._add_chimney(root, z_eave)

        return root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _openings_for(
        self,
        edge_i: int,
        v0: tuple[float, float],
        v1: tuple[float, float],
        floor_i: int,
    ) -> list[OpeningSpec]:
        p = self._p
        wall_w = math.sqrt((v1[0]-v0[0])**2 + (v1[1]-v0[1])**2)
        openings: list[OpeningSpec] = []

        if edge_i == p.front_edge and floor_i == 0:
            door = p.door
            door_cx = wall_w / 2
            openings.append(OpeningSpec(
                kind="door", center_x=door_cx, bottom_z=0.0,
                width=door.width, height=door.height, recess=door.recess, color=door.color,
            ))
            win = p.window
            min_gap = 0.4
            avail_l = door_cx - door.width / 2 - min_gap
            avail_r = wall_w - (door_cx + door.width / 2) - min_gap
            if avail_l >= win.width + min_gap * 2:
                openings.append(OpeningSpec(
                    kind="window", center_x=(door_cx - door.width / 2 - min_gap) / 2,
                    bottom_z=win.sill_height, width=win.width, height=win.height,
                    recess=win.recess, color=win.color,
                ))
            if avail_r >= win.width + min_gap * 2:
                openings.append(OpeningSpec(
                    kind="window", center_x=door_cx + door.width / 2 + min_gap + avail_r / 2,
                    bottom_z=win.sill_height, width=win.width, height=win.height,
                    recess=win.recess, color=win.color,
                ))
            return openings

        win = p.window
        n_win = min(p.windows_per_wall, max(1, int(wall_w / 2.2)))
        spacing = wall_w / (n_win + 1)
        for i in range(1, n_win + 1):
            openings.append(OpeningSpec(
                kind="window", center_x=spacing * i, bottom_z=win.sill_height,
                width=win.width, height=win.height, recess=win.recess, color=win.color,
            ))
        return openings


@register_building("l_shaped_house")
class LShapedHouse(BaseHouse):
    """
    L-shaped residential building with chimney, AC units, and balcony.

    Shares the polygon floor builder with PolygonHouse; overrides chimney
    placement to use main_width × main_depth, and adds AC unit / balcony
    support via cardinal face names mapped to the L-shape geometry.
    """

    name = "LShapedHouse"

    def __init__(self, params: LShapedHouseParams | None = None) -> None:
        self._p = params or LShapedHouseParams()

    # ------------------------------------------------------------------
    # BaseHouse abstract implementation
    # ------------------------------------------------------------------

    def _chimney_xy(self) -> tuple[float, float]:
        p = self._p
        ch = p.chimney
        return ch.pos_x * p.main_width, ch.pos_y * p.main_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, parent: NodePath | None = None) -> NodePath:
        p = self._p
        root = NodePath("l_shaped_house")
        if parent is not None:
            root.reparentTo(parent)

        verts = l_shape_verts(p.main_width, p.main_depth, p.notch_width, p.notch_depth)

        build_polygon_floors(
            root,
            verts=verts,
            num_floors=p.num_floors,
            floor_height=p.floor_height,
            floor_color=p.floor_color,
            wall_color=p.wall_color,
            openings_fn=self._openings_for,
        )

        z_eave = p.num_floors * p.floor_height
        PolygonSlab(verts, z_eave, p.roof_color, "roof").build().reparentTo(root)

        self._add_chimney(root, z_eave)

        add_ac_units(root, p.ac_units, self._ac_xy)

        if p.balcony is not None:
            bp = p.balcony
            bcx, bcy = self._balcony_xy(bp)
            add_balcony(root, bp, p.floor_height, p.num_floors, bcx, bcy)

        return root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _openings_for(
        self,
        edge_i: int,
        v0: tuple[float, float],
        v1: tuple[float, float],
        floor_i: int,
    ) -> list[OpeningSpec]:
        p = self._p
        wall_w = math.sqrt((v1[0]-v0[0])**2 + (v1[1]-v0[1])**2)
        openings: list[OpeningSpec] = []
        face = _L_EDGE_FACE.get(edge_i)

        bp = p.balcony
        if bp is not None and face == bp.face and floor_i >= bp.floor:
            door_h = min(bp.door_height, p.floor_height - 0.15)
            openings.append(OpeningSpec(
                kind="door", center_x=bp.pos_x * wall_w, bottom_z=0.0,
                width=bp.door_width, height=door_h, recess=0.05, color=bp.door_color,
            ))
            return openings

        if edge_i == 0 and floor_i == 0:
            door = p.door
            door_cx = wall_w / 2
            openings.append(OpeningSpec(
                kind="door", center_x=door_cx, bottom_z=0.0,
                width=door.width, height=door.height, recess=door.recess, color=door.color,
            ))
            win = p.window
            min_gap = 0.4
            avail_l = door_cx - door.width / 2 - min_gap
            avail_r = wall_w - (door_cx + door.width / 2) - min_gap
            if avail_l >= win.width + min_gap * 2:
                openings.append(OpeningSpec(
                    kind="window", center_x=(door_cx - door.width / 2 - min_gap) / 2,
                    bottom_z=win.sill_height, width=win.width, height=win.height,
                    recess=win.recess, color=win.color,
                ))
            if avail_r >= win.width + min_gap * 2:
                openings.append(OpeningSpec(
                    kind="window", center_x=door_cx + door.width / 2 + min_gap + avail_r / 2,
                    bottom_z=win.sill_height, width=win.width, height=win.height,
                    recess=win.recess, color=win.color,
                ))
            return openings

        win = p.window
        n_win = min(p.windows_per_wall, max(1, int(wall_w / 2.2)))
        spacing = wall_w / (n_win + 1)
        for i in range(1, n_win + 1):
            openings.append(OpeningSpec(
                kind="window", center_x=spacing * i, bottom_z=win.sill_height,
                width=win.width, height=win.height, recess=win.recess, color=win.color,
            ))
        return openings

    def _ac_xy(self, ac_p: ACUnitParams) -> tuple[float, float]:
        p = self._p
        face = ac_p.face
        if face == "south":
            return ac_p.pos_x * p.main_width, 0.0
        if face == "east":
            return p.main_width, ac_p.pos_x * (p.main_depth - p.notch_depth)
        if face == "north":
            return ac_p.pos_x * (p.main_width - p.notch_width), p.main_depth
        return 0.0, ac_p.pos_x * p.main_depth  # west

    def _balcony_xy(self, bp: BalconyParams) -> tuple[float, float]:
        p = self._p
        face = bp.face
        if face == "south":
            return bp.pos_x * p.main_width, 0.0
        if face == "east":
            return p.main_width, bp.pos_x * (p.main_depth - p.notch_depth)
        if face == "north":
            return bp.pos_x * (p.main_width - p.notch_width), p.main_depth
        return 0.0, bp.pos_x * p.main_depth  # west
