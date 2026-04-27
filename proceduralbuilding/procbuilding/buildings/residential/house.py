from __future__ import annotations

from panda3d.core import LPoint3f, NodePath

from procbuilding.buildings.residential.base_house import BaseHouse, add_ac_units, add_balcony
from procbuilding.buildings.residential.params import ResidentialHouseParams
from procbuilding.components.floor_slab import FloorSlab
from procbuilding.components.roof import RoofComponent
from procbuilding.components.wall import OpeningSpec, WallPanel
from procbuilding.registry import register_building


@register_building("residential_house")
class ResidentialHouse(BaseHouse):
    """Procedural rectangular house — walls, windows, door, gable/hip/flat roof."""

    name = "ResidentialHouse"

    def __init__(self, params: ResidentialHouseParams | None = None) -> None:
        self._p = params or ResidentialHouseParams()

    # ------------------------------------------------------------------
    # BaseHouse abstract implementation
    # ------------------------------------------------------------------

    def _chimney_xy(self) -> tuple[float, float]:
        p = self._p
        ch = p.chimney
        return ch.pos_x * p.width, ch.pos_y * p.depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, parent: NodePath | None = None) -> NodePath:
        p = self._p
        root = NodePath("residential_house")
        if parent is not None:
            root.reparentTo(parent)

        for floor_i in range(p.num_floors):
            self._build_floor(root, floor_i)

        z_eave = p.num_floors * p.floor_height
        roof = RoofComponent(
            p.roof_type,
            x0=0, y0=0, x1=p.width, y1=p.depth,
            z_eave=z_eave,
            pitch=p.roof_pitch,
            overhang=p.roof_overhang,
            color=p.roof_color,
        )
        roof.build().reparentTo(root)

        self._add_chimney(root, z_eave)

        add_ac_units(root, p.ac_units, self._ac_xy)

        if p.balcony is not None:
            bp = p.balcony
            wall_w = p.width if bp.face in ("south", "north") else p.depth
            if bp.face == "south":
                bcx, bcy = bp.pos_x * wall_w, 0.0
            elif bp.face == "north":
                bcx, bcy = bp.pos_x * wall_w, p.depth
            elif bp.face == "east":
                bcx, bcy = p.width, bp.pos_x * wall_w
            else:  # west
                bcx, bcy = 0.0, bp.pos_x * wall_w
            add_balcony(root, bp, p.floor_height, p.num_floors, bcx, bcy)

        return root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ac_xy(self, ac_p) -> tuple[float, float]:
        p = self._p
        face = ac_p.face
        if face in ("south", "north"):
            cx = ac_p.pos_x * p.width
            cy = 0.0 if face == "south" else p.depth
        else:
            cx = p.width if face == "east" else 0.0
            cy = ac_p.pos_x * p.depth
        return cx, cy

    def _build_floor(self, root: NodePath, floor_i: int) -> None:
        p = self._p
        z0 = floor_i * p.floor_height
        floor_node = root.attachNewNode(f"floor_{floor_i}")

        slab = FloorSlab(0, 0, z0, p.width, p.depth, p.floor_color, f"slab_{floor_i}")
        slab.build().reparentTo(floor_node)

        faces = {
            "south": (LPoint3f(0,       0,       z0), p.width),
            "north": (LPoint3f(p.width, p.depth, z0), p.width),
            "west":  (LPoint3f(0,       p.depth, z0), p.depth),
            "east":  (LPoint3f(p.width, 0,       z0), p.depth),
        }
        for face, (origin, wall_width) in faces.items():
            openings = self._openings_for(face, floor_i, wall_width)
            WallPanel(
                origin=origin,
                width=wall_width,
                height=p.floor_height,
                face=face,
                color=p.wall_color,
                openings=openings,
            ).build().reparentTo(floor_node)

    def _openings_for(
        self, face: str, floor_i: int, wall_width: float
    ) -> list[OpeningSpec]:
        p = self._p
        openings: list[OpeningSpec] = []

        is_front = face == p.front_face
        is_ground = floor_i == 0

        cap = p.windows_per_long_wall if face in ("south", "north") else p.windows_per_short_wall

        bp = p.balcony
        if bp is not None and bp.face == face and floor_i >= bp.floor:
            door_h = min(bp.door_height, p.floor_height - 0.15)
            openings.append(OpeningSpec(
                kind="door",
                center_x=bp.pos_x * wall_width,
                bottom_z=0.0,
                width=bp.door_width,
                height=door_h,
                recess=0.05,
                color=bp.door_color,
            ))
            return openings

        if is_front and is_ground:
            door = p.door
            door_cx = wall_width / 2
            openings.append(OpeningSpec(
                kind="door",
                center_x=door_cx,
                bottom_z=0.0,
                width=door.width,
                height=door.height,
                recess=door.recess,
                color=door.color,
            ))
            win = p.window
            min_gap = 0.4
            avail_left = door_cx - door.width / 2 - min_gap
            avail_right = wall_width - (door_cx + door.width / 2) - min_gap
            if avail_left >= win.width + min_gap * 2:
                cx = (door_cx - door.width / 2 - min_gap) / 2
                openings.append(OpeningSpec(
                    kind="window",
                    center_x=cx,
                    bottom_z=win.sill_height,
                    width=win.width,
                    height=win.height,
                    recess=win.recess,
                    color=win.color,
                ))
            if avail_right >= win.width + min_gap * 2:
                cx = door_cx + door.width / 2 + min_gap + avail_right / 2
                openings.append(OpeningSpec(
                    kind="window",
                    center_x=cx,
                    bottom_z=win.sill_height,
                    width=win.width,
                    height=win.height,
                    recess=win.recess,
                    color=win.color,
                ))
            return openings

        win = p.window
        # Density-based count: target ~2 m centre-to-centre, capped by param
        n_win = min(cap, max(1, int(wall_width / 2.2)))
        spacing = wall_width / (n_win + 1)
        for i in range(1, n_win + 1):
            openings.append(OpeningSpec(
                kind="window",
                center_x=spacing * i,
                bottom_z=win.sill_height,
                width=win.width,
                height=win.height,
                recess=win.recess,
                color=win.color,
            ))
        return openings
