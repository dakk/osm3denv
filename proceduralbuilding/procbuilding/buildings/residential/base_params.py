"""Shared parameter base for all residential building variants."""
from __future__ import annotations

from dataclasses import dataclass, field

from procbuilding.buildings.residential.component_params import (
    ChimneyParams,
    DoorParams,
    WindowParams,
)
from procbuilding.params import Color


@dataclass
class BaseHouseParams:
    """
    Fields shared by every residential building variant.

    Subclasses add footprint geometry (width×depth or polygon verts),
    roof configuration, and any decorations specific to their shape.
    """

    floor_height: float = 3.0
    num_floors: int = 2

    wall_color: Color = (0.9, 0.85, 0.75, 1.0)
    floor_color: Color = (0.6, 0.5, 0.4, 1.0)
    roof_color: Color = (0.5, 0.25, 0.1, 1.0)

    window: WindowParams = field(default_factory=WindowParams)
    door: DoorParams = field(default_factory=DoorParams)

    chimney: ChimneyParams | None = None
