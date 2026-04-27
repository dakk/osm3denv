"""Parameters for polygon-footprint and L-shaped residential buildings."""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

from procbuilding.buildings.residential.base_params import BaseHouseParams
from procbuilding.buildings.residential.component_params import (
    ACUnitParams,
    BalconyParams,
    ChimneyParams,
)
from procbuilding.geometry.polygon import Vert2D, ensure_ccw, l_shape_verts


@dataclass
class PolygonHouseParams(BaseHouseParams):
    """
    Residential building with an arbitrary simple-polygon footprint.

    verts: CCW list of (x, y) tuples — auto-normalised in __post_init__.
    Chimney pos_x / pos_y are fractions of the polygon's bounding box.
    Roof is always flat; AC units and balcony are not supported on arbitrary
    polygons (use LShapedHouseParams for those).
    """

    verts: list[Vert2D] = field(default_factory=list)
    windows_per_wall: int = 8
    front_edge: int = 0

    def __post_init__(self) -> None:
        if not self.verts:
            raise ValueError("PolygonHouseParams: verts must be non-empty")
        self.verts = ensure_ccw(list(self.verts))
        if not (0 <= self.front_edge < len(self.verts)):
            raise ValueError(
                f"front_edge={self.front_edge} out of range "
                f"for {len(self.verts)}-vertex polygon"
            )


@dataclass
class LShapedHouseParams(BaseHouseParams):
    """
    L-shaped building: main_width × main_depth rectangle with a
    notch_width × notch_depth notch cut from the back-right corner.

    Layout (Y increases northward):

        (0, md)-----(mw-nw, md)
           |              |
           |         (mw-nw, md-nd)---(mw, md-nd)
           |                               |
        (0,  0)----------------(mw, 0)

    Balcony and AC units reference outer faces by cardinal name:
      "south"  length = main_width,               at y = 0
      "east"   length = main_depth − notch_depth, at x = main_width
      "north"  length = main_width − notch_width, at y = main_depth
      "west"   length = main_depth,               at x = 0
    """

    main_width: float = 10.0
    main_depth: float = 8.0
    notch_width: float = 4.0
    notch_depth: float = 3.0

    # Override chimney default: L-shaped houses ship with one by default
    chimney: ChimneyParams | None = field(default_factory=ChimneyParams)

    ac_units: list[ACUnitParams] = field(default_factory=lambda: [ACUnitParams()])
    balcony: BalconyParams | None = None

    windows_per_wall: int = 8

    def __post_init__(self) -> None:
        if self.notch_width >= self.main_width:
            raise ValueError("LShapedHouseParams: notch_width must be < main_width")
        if self.notch_depth >= self.main_depth:
            raise ValueError("LShapedHouseParams: notch_depth must be < main_depth")
        if self.balcony is not None and self.balcony.floor < 1:
            raise ValueError("LShapedHouseParams: balcony.floor must be >= 1")

    def outer_wall_width(self, face: str) -> float:
        if face == "south":
            return self.main_width
        if face == "east":
            return self.main_depth - self.notch_depth
        if face == "north":
            return self.main_width - self.notch_width
        return self.main_depth  # west

    def to_polygon_params(self) -> PolygonHouseParams:
        """Produce a PolygonHouseParams with the L-shape polygon and no AC/balcony."""
        return PolygonHouseParams(
            verts=l_shape_verts(
                self.main_width, self.main_depth,
                self.notch_width, self.notch_depth,
            ),
            floor_height=self.floor_height,
            num_floors=self.num_floors,
            wall_color=self.wall_color,
            floor_color=self.floor_color,
            roof_color=self.roof_color,
            window=self.window,
            door=self.door,
            chimney=self.chimney,
            windows_per_wall=self.windows_per_wall,
            front_edge=0,
        )

    @classmethod
    def random(cls, seed: int | None = None) -> "LShapedHouseParams":
        rng = _random.Random(seed)

        main_width = round(rng.uniform(10.0, 16.0), 1)
        main_depth = round(rng.uniform(8.0, 13.0), 1)
        notch_width = round(rng.uniform(main_width * 0.30, main_width * 0.55), 1)
        notch_depth = round(rng.uniform(main_depth * 0.30, main_depth * 0.55), 1)
        num_floors = rng.randint(1, 3)

        chimney: ChimneyParams | None = None
        if rng.random() < 0.7:
            chimney = ChimneyParams(
                pos_x=round(rng.uniform(0.2, 0.8), 2),
                pos_y=round(rng.uniform(0.25, 0.75), 2),
                height=round(rng.uniform(0.5, 2.0), 1),
            )

        n_ac = rng.randint(0, 2)
        ac_faces = rng.sample(["south", "east", "north", "west"], k=min(n_ac, 4))
        ac_units = [
            ACUnitParams(face=face, pos_x=round(rng.uniform(0.2, 0.8), 2))
            for face in ac_faces
        ]

        balcony: BalconyParams | None = None
        if num_floors >= 2 and rng.random() < 0.5:
            b_face = rng.choice(["south", "east", "north", "west"])
            wall_w = _outer_wall_width(b_face, main_width, main_depth, notch_width, notch_depth)
            pos_x = round(rng.uniform(0.25, 0.75), 2)
            max_hw = min(pos_x, 1.0 - pos_x) * wall_w
            b_width = round(min(rng.uniform(1.5, 3.0), wall_w * 0.65, max_hw * 2), 1)
            balcony = BalconyParams(
                face=b_face,
                floor=rng.randint(1, num_floors - 1),
                width=b_width,
                protrusion=round(rng.uniform(0.8, 1.5), 1),
                pos_x=pos_x,
            )

        return cls(
            main_width=main_width,
            main_depth=main_depth,
            notch_width=notch_width,
            notch_depth=notch_depth,
            num_floors=num_floors,
            windows_per_wall=rng.randint(5, 8),
            chimney=chimney,
            ac_units=ac_units,
            balcony=balcony,
        )


def _outer_wall_width(face: str, mw: float, md: float, nw: float, nd: float) -> float:
    if face == "south":
        return mw
    if face == "east":
        return md - nd
    if face == "north":
        return mw - nw
    return md
