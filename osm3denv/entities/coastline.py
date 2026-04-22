"""Coastline entity — OSM natural=coastline ways rendered as polylines."""
from __future__ import annotations

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame


class Coastline(MapEntity):
    COLOR = (1.0, 0.85, 0.2, 1.0)

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, sea_z: float = 0.0) -> None:
        self._osm = osm
        self._frame = frame
        self._radius_m = radius_m
        self._z = sea_z + 0.5   # sit just above the sea surface
        self._polylines: list[np.ndarray] = []

    def build(self) -> None:
        ways = self._osm.filter_ways(lambda t: t.get("natural") == "coastline")
        r = float(self._radius_m)
        z = self._z

        for way in ways:
            if len(way.geometry) < 2:
                continue
            lons = np.fromiter((g[0] for g in way.geometry), dtype=np.float64,
                               count=len(way.geometry))
            lats = np.fromiter((g[1] for g in way.geometry), dtype=np.float64,
                               count=len(way.geometry))
            east, north = self._frame.to_enu(lons, lats)
            inside = (np.abs(east) <= r) & (np.abs(north) <= r)
            segment: list[list[float]] = []
            for i in range(len(east)):
                if inside[i]:
                    segment.append([float(east[i]), float(north[i]), z])
                elif segment:
                    if len(segment) >= 2:
                        self._polylines.append(np.array(segment, dtype=np.float32))
                    segment = []
            if len(segment) >= 2:
                self._polylines.append(np.array(segment, dtype=np.float32))

    def attach_to(self, parent) -> None:
        if not self._polylines:
            return
        from osm3denv.render.helpers import attach_lines
        np_ = attach_lines(parent, "coastline", self._polylines, self.COLOR, thickness=2.0)
        np_.setLightOff()
