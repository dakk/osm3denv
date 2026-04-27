from enum import Enum, auto
from typing import Tuple

Color = Tuple[float, float, float, float]


class RoofType(Enum):
    FLAT = auto()
    GABLE = auto()
    HIP = auto()
