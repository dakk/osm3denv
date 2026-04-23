"""Power-line entity — transmission lines and distribution poles.

Cables are draped over the terrain with a parabolic sag between supports.
High-voltage towers get a cross-arm structure oriented perpendicular to the
line direction.  Distribution poles are rendered as simple vertical sticks.
"""
from __future__ import annotations

import logging

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.entities.utils import sample_z
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_TOWER_HEIGHT: dict[str, float] = {
    "line":       22.0,
    "minor_line":  8.0,
}

# Cross-arm half-length (metres from mast centre to cable attachment)
_ARM_LEN: dict[str, float] = {
    "line":       5.5,   # lattice transmission tower
    "minor_line": 0.0,   # distribution pole — no arms
}

_SAG_RATIO  = 0.02   # mid-span sag as fraction of span length
_SAG_STEPS  = 8

_CABLE_COLOR     = (0.45, 0.45, 0.45, 1.0)
_STRUCTURE_COLOR = (0.55, 0.55, 0.55, 1.0)


def _tower_segments(e: float, n: float, terrain_z: float,
                    h: float, arm: float,
                    perp_x: float, perp_y: float) -> list[np.ndarray]:
    """Return line segments that form a transmission tower structure."""
    top  = terrain_z + h
    mid  = terrain_z + h * 0.75   # lower cross-arm elevation

    segs: list[np.ndarray] = []

    # Vertical mast
    segs.append(np.array([[e, n, terrain_z], [e, n, top]], dtype=np.float32))

    if arm > 0.0:
        # Upper cross-arm
        segs.append(np.array([
            [e - perp_x * arm, n - perp_y * arm, top],
            [e + perp_x * arm, n + perp_y * arm, top],
        ], dtype=np.float32))

        # Lower cross-arm (narrower — carries the earth/guard wire)
        arm2 = arm * 0.55
        segs.append(np.array([
            [e - perp_x * arm2, n - perp_y * arm2, mid],
            [e + perp_x * arm2, n + perp_y * arm2, mid],
        ], dtype=np.float32))

        # Diagonal braces from mast to upper arm tips (gives lattice feel)
        brace_z = terrain_z + h * 0.88
        for sign in (1.0, -1.0):
            segs.append(np.array([
                [e + sign * perp_x * arm, n + sign * perp_y * arm, top],
                [e, n, brace_z],
            ], dtype=np.float32))

    return segs


class PowerLines(MapEntity):
    """High-voltage lines and distribution poles draped over the terrain."""

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, terrain) -> None:
        self._osm      = osm
        self._frame    = frame
        self._radius_m = radius_m
        self._terrain  = terrain
        self._cables:     list[np.ndarray] = []
        self._structures: list[np.ndarray] = []   # masts + cross-arms

    def build(self) -> None:
        td        = self._terrain.data
        heightmap = td.heightmap
        grid      = heightmap.shape[0]
        r         = float(self._radius_m)

        def z_at(e, n):
            return sample_z(float(e), float(n), heightmap, grid, r)

        n_ways = 0
        for way in self._osm.filter_ways(
                lambda t: t.get("power") in _TOWER_HEIGHT):
            geom = way.geometry
            if len(geom) < 2:
                continue

            lons = np.fromiter((p[0] for p in geom), np.float64, count=len(geom))
            lats = np.fromiter((p[1] for p in geom), np.float64, count=len(geom))
            east, north = self._frame.to_enu(lons, lats)

            if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                continue

            ptype  = way.tags["power"]
            h      = _TOWER_HEIGHT[ptype]
            arm    = _ARM_LEN[ptype]
            pts: list[list[float]] = []

            for i in range(len(east)):
                e, n = float(east[i]), float(north[i])
                in_scene = abs(e) <= r and abs(n) <= r
                terrain_z = z_at(e, n) if in_scene else 0.0

                if not in_scene:
                    if len(pts) >= 2:
                        self._cables.append(np.array(pts, dtype=np.float32))
                    pts = []
                    continue

                attach_z = terrain_z + h

                # --- Tower / pole structure at this support point ---
                # Perpendicular direction: average of incoming and outgoing
                # segment directions, rotated 90°.
                dx, dy = 0.0, 0.0
                if i > 0:
                    dx += float(east[i]  - east[i - 1])
                    dy += float(north[i] - north[i - 1])
                if i < len(east) - 1:
                    dx += float(east[i + 1]  - east[i])
                    dy += float(north[i + 1] - north[i])
                seg_len = max(1e-8, (dx * dx + dy * dy) ** 0.5)
                # Perpendicular: rotate 90° → (−dy, dx)
                perp_x, perp_y = -dy / seg_len, dx / seg_len

                self._structures.extend(
                    _tower_segments(e, n, terrain_z, h, arm, perp_x, perp_y))

                # --- Cable span with parabolic sag ---
                if pts:
                    pe, pn, pz = pts[-1]
                    span = float(np.hypot(e - pe, n - pn))
                    sag  = span * _SAG_RATIO
                    for s in range(1, _SAG_STEPS + 1):
                        t  = s / (_SAG_STEPS + 1)
                        ie  = pe + t * (e - pe)
                        in_ = pn + t * (n - pn)
                        iz  = pz + t * (attach_z - pz) - sag * 4.0 * t * (1.0 - t)
                        pts.append([ie, in_, iz])

                pts.append([e, n, attach_z])

            if len(pts) >= 2:
                self._cables.append(np.array(pts, dtype=np.float32))
            n_ways += 1

        log.info("powerlines: %d ways → %d cable spans, %d structure segments",
                 n_ways, len(self._cables), len(self._structures))

    def attach_to(self, parent) -> None:
        from osm3denv.render.helpers import attach_lines

        if self._cables:
            np_ = attach_lines(parent, "power_cables", self._cables,
                                _CABLE_COLOR, thickness=0.8)
            np_.setLightOff()

        if self._structures:
            np_ = attach_lines(parent, "power_structures", self._structures,
                                _STRUCTURE_COLOR, thickness=1.5)
            np_.setLightOff()
