"""Per-element parameter dataclasses (window, door, chimney, AC unit, balcony)."""
from __future__ import annotations

import random as _random
from dataclasses import dataclass

from procbuilding.params import Color


@dataclass
class WindowParams:
    width: float = 0.8
    height: float = 1.0
    recess: float = 0.08
    sill_height: float = 0.9   # distance from floor level to window bottom
    color: Color = (0.4, 0.6, 0.9, 1.0)


@dataclass
class DoorParams:
    width: float = 0.9
    height: float = 2.1
    recess: float = 0.08
    color: Color = (0.4, 0.25, 0.1, 1.0)


@dataclass
class ChimneyParams:
    width: float = 0.5
    depth: float = 0.5
    height: float = 2.0        # above z_eave
    pos_x: float = 0.3         # 0-1 fraction of building width (or bbox width)
    pos_y: float = 0.35        # 0-1 fraction of building depth (or bbox depth)
    color: Color = (0.48, 0.40, 0.32, 1.0)
    cap_color: Color = (0.28, 0.28, 0.28, 1.0)


@dataclass
class ACUnitParams:
    width: float = 0.85
    depth: float = 0.30
    height: float = 0.55
    pos_x: float = 0.25        # 0-1 fraction along the chosen wall
    face: str = "east"         # wall it is mounted on
    z_bottom: float = 0.15     # height off the ground
    color: Color = (0.78, 0.78, 0.80, 1.0)
    grille_color: Color = (0.50, 0.50, 0.53, 1.0)


@dataclass
class BalconyParams:
    width: float = 2.0          # span along the wall face
    protrusion: float = 1.0     # how far it sticks out from the wall
    railing_height: float = 0.9 # height of the railing above the balcony floor
    pos_x: float = 0.5          # 0-1 fraction along the chosen wall face
    face: str = "south"         # wall the balcony is attached to
    floor: int = 1              # 0-indexed storey (must be >= 1, i.e. not ground)
    door_width: float = 0.9     # balcony access door width
    door_height: float = 2.1    # balcony access door height (clamped to floor_height - 0.15)
    color: Color = (0.88, 0.85, 0.78, 1.0)
    railing_color: Color = (0.80, 0.78, 0.72, 1.0)
    door_color: Color = (0.35, 0.22, 0.10, 1.0)

    def __post_init__(self) -> None:
        if self.floor < 1:
            raise ValueError(f"BalconyParams.floor must be >= 1 (got {self.floor})")
