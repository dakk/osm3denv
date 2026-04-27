from abc import ABC, abstractmethod
from panda3d.core import NodePath


class BuildingComponent(ABC):
    """A single geometric unit of a building."""

    @abstractmethod
    def build(self) -> NodePath:
        """Construct and return geometry as an unattached NodePath."""
