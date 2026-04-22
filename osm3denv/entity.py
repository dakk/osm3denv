"""Base class for all map entities.

A ``MapEntity`` owns its full lifecycle: it fetches/computes geometry in
``build()`` and attaches Panda3D nodes in ``attach_to()``.  The renderer only
needs to call these two methods — it does not need to know what any entity
contains.
"""
from __future__ import annotations


class MapEntity:
    def build(self) -> None:
        raise NotImplementedError(f"{type(self).__name__}.build()")

    def attach_to(self, parent) -> None:
        raise NotImplementedError(f"{type(self).__name__}.attach_to()")
