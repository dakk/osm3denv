"""Parameters for the rectangular (quad) residential house."""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

from procbuilding.buildings.residential.base_params import BaseHouseParams
from procbuilding.buildings.residential.component_params import (  # re-exported for compat
    ACUnitParams,
    BalconyParams,
    ChimneyParams,
    DoorParams,
    WindowParams,
)
from procbuilding.params import Color, RoofType


@dataclass
class ResidentialHouseParams(BaseHouseParams):
    """
    Parameters for a rectangular residential house.

    Inherits shared fields (colors, floor dims, window, door, chimney) from
    BaseHouseParams and adds rectangle-specific geometry and decoration.
    """

    # Footprint
    width: float = 10.0
    depth: float = 8.0

    # Roof
    roof_type: RoofType = RoofType.GABLE
    roof_pitch: float = 0.5
    roof_overhang: float = 0.4

    # Override chimney default: rectangular houses ship with one by default
    chimney: ChimneyParams | None = field(default_factory=ChimneyParams)

    # Decorations (cardinal-face-based — not meaningful for polygon buildings)
    ac_units: list[ACUnitParams] = field(default_factory=lambda: [ACUnitParams()])
    balcony: BalconyParams | None = None

    # Window layout
    windows_per_long_wall: int = 8
    windows_per_short_wall: int = 5
    front_face: str = "south"

    @classmethod
    def random(cls, seed: int | None = None) -> "ResidentialHouseParams":
        """Return a random but architecturally plausible configuration."""
        rng = _random.Random(seed)

        width       = round(rng.uniform(8.0, 16.0), 1)
        depth       = round(rng.uniform(6.0, 11.0), 1)
        floor_height= round(rng.uniform(2.8, 3.4),  1)
        num_floors  = rng.randint(1, 3)
        roof_type   = rng.choice(list(RoofType))
        roof_pitch  = round(rng.uniform(0.3, 0.75), 2)
        roof_overhang = round(rng.uniform(0.2, 0.6), 1)

        chimney: ChimneyParams | None = None
        if rng.random() < 0.7:
            pos_x = round(rng.uniform(0.2, 0.8), 2)
            pos_y = round(rng.uniform(0.25, 0.45), 2)
            dist_from_ridge = abs(pos_y * depth - depth / 2)
            roof_above_eave = roof_pitch * (depth / 2 - dist_from_ridge)
            min_height = roof_above_eave + 0.3
            height = max(round(rng.uniform(1.5, 3.0), 1), round(min_height, 1))
            chimney = ChimneyParams(pos_x=pos_x, pos_y=pos_y, height=height)

        n_ac = rng.randint(0, 2)
        ac_faces = rng.sample(["south", "north", "east", "west"], k=min(n_ac, 4))
        ac_units = [
            ACUnitParams(face=face, pos_x=round(rng.uniform(0.2, 0.8), 2))
            for face in ac_faces
        ]

        balcony: BalconyParams | None = None
        if num_floors >= 2 and rng.random() < 0.5:
            b_face = rng.choice(["south", "north", "east", "west"])
            wall_w = width if b_face in ("south", "north") else depth
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
            width=width,
            depth=depth,
            floor_height=floor_height,
            num_floors=num_floors,
            roof_type=roof_type,
            roof_pitch=roof_pitch,
            roof_overhang=roof_overhang,
            windows_per_long_wall=rng.randint(5, 8),
            windows_per_short_wall=rng.randint(3, 5),
            chimney=chimney,
            ac_units=ac_units,
            balcony=balcony,
        )
