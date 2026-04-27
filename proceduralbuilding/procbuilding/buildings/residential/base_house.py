"""Abstract base class for residential buildings and shared polygon-floor builder."""
from __future__ import annotations

from abc import ABC, abstractmethod

from panda3d.core import NodePath

from procbuilding.buildings.base import Building
from procbuilding.buildings.residential.base_params import BaseHouseParams
from procbuilding.buildings.residential.component_params import ACUnitParams, BalconyParams
from procbuilding.components.ac_unit import ACUnit
from procbuilding.components.balcony import BalconyComponent
from procbuilding.components.chimney import ChimneyComponent
from procbuilding.components.edge_wall import EdgeWallPanel
from procbuilding.components.polygon_slab import PolygonSlab
from procbuilding.components.wall import OpeningSpec
from procbuilding.geometry.polygon import Vert2D, l_shape_verts


class BaseHouse(Building, ABC):
    """
    Shared skeleton for all residential building classes.

    Subclasses must implement :meth:`build` and :meth:`_chimney_xy`.
    They call :meth:`_add_chimney` inside their own ``build`` to place the
    chimney without duplicating the ChimneyComponent wiring.

    AC unit and balcony placement are geometry-specific and are handled by
    helper methods in this module rather than abstract methods, to avoid
    forcing non-supporting subclasses to stub them out.
    """

    _p: BaseHouseParams

    @abstractmethod
    def build(self, parent: NodePath | None = None) -> NodePath:
        ...

    @abstractmethod
    def _chimney_xy(self) -> tuple[float, float]:
        """Return world (x, y) centre for chimney placement."""

    def _add_chimney(self, root: NodePath, z_eave: float) -> None:
        ch = self._p.chimney
        if ch is None:
            return
        cx, cy = self._chimney_xy()
        ChimneyComponent(
            x_center=cx,
            y_center=cy,
            width=ch.width,
            depth=ch.depth,
            z_eave=z_eave,
            height=ch.height,
            color=ch.color,
            cap_color=ch.cap_color,
        ).build().reparentTo(root)


# ---------------------------------------------------------------------------
# Shared decoration helpers (AC units and balcony are geometry-specific so
# they use callable arguments for position rather than virtual methods)
# ---------------------------------------------------------------------------

def add_ac_units(
    root: NodePath,
    ac_units: list[ACUnitParams],
    xy_fn,          # (ACUnitParams) -> (float, float)
) -> None:
    """Place AC units on *root* using *xy_fn* to compute world (x, y)."""
    for ac in ac_units:
        cx, cy = xy_fn(ac)
        ACUnit(
            cx=cx,
            cy=cy,
            z_bottom=ac.z_bottom,
            width=ac.width,
            depth=ac.depth,
            height=ac.height,
            face=ac.face,
            color=ac.color,
            grille_color=ac.grille_color,
        ).build().reparentTo(root)


def add_balcony(
    root: NodePath,
    bp: BalconyParams,
    floor_height: float,
    num_floors: int,
    cx: float,
    cy: float,
) -> None:
    """Place balcony slabs and railings on *root* for every floor >= bp.floor."""
    for floor_i in range(bp.floor, num_floors):
        BalconyComponent(
            cx=cx,
            cy=cy,
            z_floor=floor_i * floor_height,
            width=bp.width,
            protrusion=bp.protrusion,
            railing_height=bp.railing_height,
            face=bp.face,
            slab_color=bp.color,
            railing_color=bp.railing_color,
        ).build().reparentTo(root)


# ---------------------------------------------------------------------------
# Shared polygon floor builder
# ---------------------------------------------------------------------------

def build_polygon_floors(
    root: NodePath,
    verts: list[Vert2D],
    num_floors: int,
    floor_height: float,
    floor_color,
    wall_color,
    openings_fn,    # (edge_i, v0, v1, floor_i) -> list[OpeningSpec]
) -> None:
    """
    Build all floor slabs and wall panels for a polygon-footprint building.

    Called by both PolygonHouse and LShapedHouse — the only difference
    between them is the *openings_fn* callback and the vertex list.
    """
    n_edges = len(verts)
    for floor_i in range(num_floors):
        z0 = floor_i * floor_height
        floor_node = root.attachNewNode(f"floor_{floor_i}")

        PolygonSlab(verts, z0, floor_color, f"slab_{floor_i}").build().reparentTo(floor_node)

        for edge_i in range(n_edges):
            v0 = verts[edge_i]
            v1 = verts[(edge_i + 1) % n_edges]
            openings = openings_fn(edge_i, v0, v1, floor_i)
            EdgeWallPanel(
                p0=v0,
                p1=v1,
                z_bottom=z0,
                height=floor_height,
                color=wall_color,
                openings=openings,
            ).build().reparentTo(floor_node)
