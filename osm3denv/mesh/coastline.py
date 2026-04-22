"""Extract OSM natural=coastline ways as ENU polylines → RenderLayer."""
from __future__ import annotations

import numpy as np

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.layer import RenderLayer


def build(osm: OSMData, frame: Frame, radius_m: float,
          z: float = 0.3) -> list[RenderLayer]:
    """Return a single ``RenderLayer`` containing all coastline polylines.

    Parameters
    ----------
    z:
        Fixed height for all coastline vertices (typically ``sea_z + 0.5``).
    """
    ways = osm.filter_ways(lambda t: t.get("natural") == "coastline")
    r = float(radius_m)
    polylines: list[np.ndarray] = []

    for way in ways:
        if len(way.geometry) < 2:
            continue
        lons = np.fromiter((g[0] for g in way.geometry), dtype=np.float64,
                           count=len(way.geometry))
        lats = np.fromiter((g[1] for g in way.geometry), dtype=np.float64,
                           count=len(way.geometry))
        east, north = frame.to_enu(lons, lats)

        inside = (np.abs(east) <= r) & (np.abs(north) <= r)
        segment: list[list[float]] = []
        for i in range(len(east)):
            if inside[i]:
                segment.append([float(east[i]), float(north[i]), z])
            elif segment:
                if len(segment) >= 2:
                    polylines.append(np.array(segment, dtype=np.float32))
                segment = []
        if len(segment) >= 2:
            polylines.append(np.array(segment, dtype=np.float32))

    if not polylines:
        return []

    return [RenderLayer(
        name="coastline",
        polylines=polylines,
        line_thickness=2.0,
        color=(1.0, 0.85, 0.2, 1.0),
        lit=False,
    )]
