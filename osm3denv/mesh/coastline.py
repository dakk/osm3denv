"""Extract OSM natural=coastline ways as ENU polylines.

The coastline is the land/sea boundary. We project each way to the local ENU
frame and keep only the portion inside the terrain square, splitting a way
into multiple polylines when it crosses the boundary. No clipping with
interpolation — we just drop out-of-bounds vertices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame


@dataclass
class CoastlineData:
    polylines: list[np.ndarray]  # each (N, 2) float32 — east, north


def build(osm: OSMData, frame: Frame, radius_m: float) -> CoastlineData:
    ways = osm.filter_ways(lambda t: t.get("natural") == "coastline")
    out: list[np.ndarray] = []
    r = float(radius_m)

    for way in ways:
        if len(way.geometry) < 2:
            continue
        lons = np.fromiter((g[0] for g in way.geometry), dtype=np.float64,
                           count=len(way.geometry))
        lats = np.fromiter((g[1] for g in way.geometry), dtype=np.float64,
                           count=len(way.geometry))
        east, north = frame.to_enu(lons, lats)
        pts = np.stack([east, north], axis=-1).astype(np.float32)

        inside = (np.abs(pts[:, 0]) <= r) & (np.abs(pts[:, 1]) <= r)
        current: list[np.ndarray] = []
        for i in range(len(pts)):
            if inside[i]:
                current.append(pts[i])
            elif current:
                if len(current) >= 2:
                    out.append(np.asarray(current, dtype=np.float32))
                current = []
        if len(current) >= 2:
            out.append(np.asarray(current, dtype=np.float32))

    return CoastlineData(polylines=out)
