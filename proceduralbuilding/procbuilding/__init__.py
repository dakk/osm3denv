from procbuilding.registry import (
    register_building,
    get_building_class,
    list_building_types,
    build,
    load_entry_point_plugins,
)
from procbuilding.buildings.base import Building
from procbuilding.buildings.residential.base_params import BaseHouseParams
from procbuilding.buildings.residential.component_params import (
    WindowParams,
    DoorParams,
    ChimneyParams,
    ACUnitParams,
    BalconyParams,
)
from procbuilding.buildings.residential.house import ResidentialHouse
from procbuilding.buildings.residential.params import ResidentialHouseParams
from procbuilding.buildings.residential.polygon_house import (  # noqa: F401 — registers types
    LShapedHouse,
    PolygonHouse,
)
from procbuilding.buildings.residential.polygon_params import (
    LShapedHouseParams,
    PolygonHouseParams,
)
from procbuilding.params import RoofType

# Auto-load installed third-party building plugins
load_entry_point_plugins()

__all__ = [
    "register_building",
    "get_building_class",
    "list_building_types",
    "build",
    "Building",
    "BaseHouseParams",
    "ResidentialHouse",
    "ResidentialHouseParams",
    "WindowParams",
    "DoorParams",
    "ChimneyParams",
    "ACUnitParams",
    "BalconyParams",
    "RoofType",
    "PolygonHouse",
    "PolygonHouseParams",
    "LShapedHouse",
    "LShapedHouseParams",
]
