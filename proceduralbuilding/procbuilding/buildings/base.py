from abc import ABC, abstractmethod
from panda3d.core import NodePath


class Building(ABC):
    """Abstract base for all building types."""

    @abstractmethod
    def build(self, parent: NodePath | None) -> NodePath:
        """Construct the building as a NodePath subtree under *parent*."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable label used for the scene graph node."""
