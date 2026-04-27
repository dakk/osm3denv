from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Type

if TYPE_CHECKING:
    from procbuilding.buildings.base import Building
    from panda3d.core import NodePath

_REGISTRY: dict[str, type["Building"]] = {}

T = TypeVar("T", bound="Building")


def register_building(name: str):
    """Class decorator — registers a Building subclass under *name*."""
    def decorator(cls: Type[T]) -> Type[T]:
        if name in _REGISTRY:
            raise ValueError(f"Building type '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_building_class(name: str) -> type["Building"]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No building type '{name}' registered. "
            f"Available: {list(_REGISTRY.keys())}"
        )


def list_building_types() -> list[str]:
    return list(_REGISTRY.keys())


def build(name: str, params=None, parent: "NodePath | None" = None) -> "NodePath":
    """Look up, instantiate, and call build() on a registered building type."""
    cls = get_building_class(name)
    instance = cls(params)
    return instance.build(parent)


def load_entry_point_plugins() -> None:
    """Auto-load packages registered under the 'procbuilding.buildings' entry-point group."""
    try:
        from importlib.metadata import entry_points
        for ep in entry_points(group="procbuilding.buildings"):
            ep.load()
    except Exception:
        pass
