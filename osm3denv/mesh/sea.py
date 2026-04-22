"""Build a sea polygon from OSM natural=coastline ways.

OSM convention: the sea is on the **right-hand side** of a coastline in node
order. We project each coastline to the local ENU frame, clip to the terrain
square, then polygonize the union of (bbox boundary + clipped coastlines) to
enumerate every closed region; the ones whose representative point falls on
the sea side of the nearest coastline segment are merged into the sea polygon.

Consumed by :mod:`osm3denv.mesh.terrain` to clamp terrain cells inside the sea
to just below absolute sea level, so the flat sea plane in the viewer fully
occludes SRTM over-ocean noise.
"""
from __future__ import annotations

import logging

import numpy as np
import shapely
import shapely.geometry as sg
import shapely.ops

from osm3denv.fetch.osm import OSMData, OSMWay
from osm3denv.frame import Frame

log = logging.getLogger(__name__)


def _coastline_to_enu(way: OSMWay, frame: Frame) -> sg.LineString | None:
    if len(way.geometry) < 2:
        return None
    lon = np.fromiter((p[0] for p in way.geometry), dtype=np.float64,
                      count=len(way.geometry))
    lat = np.fromiter((p[1] for p in way.geometry), dtype=np.float64,
                      count=len(way.geometry))
    east, north = frame.to_enu(lon, lat)
    coords = list(zip(east, north))
    line = sg.LineString(coords)
    return line if line.length > 0.5 else None


def _on_sea_side(poly: sg.Polygon, coast_lines: list[sg.LineString]) -> bool:
    """True iff the polygon's interior lies on the right-hand side (sea side)
    of the nearest coastline segment across all coastline lines."""
    p = poly.representative_point()
    px, py = p.x, p.y
    best_d = float("inf")
    best_cross = 0.0
    for line in coast_lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            ax, ay = coords[i]
            bx, by = coords[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= 1e-12:
                continue
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
            nx, ny = ax + t * dx, ay + t * dy
            d = float(np.hypot(px - nx, py - ny))
            if d < best_d:
                best_d = d
                best_cross = dx * (py - ay) - dy * (px - ax)
    return best_cross < 0.0


def build_sea_polygon(osm: OSMData, frame: Frame,
                      radius_m: float) -> sg.Polygon | sg.MultiPolygon | None:
    coast_ways = osm.filter_ways(lambda t: t.get("natural") == "coastline")
    if not coast_ways:
        return None

    bbox = sg.box(-radius_m, -radius_m, radius_m, radius_m)
    coast_lines: list[sg.LineString] = []
    for w in coast_ways:
        line = _coastline_to_enu(w, frame)
        if line is None:
            continue
        clipped = line.intersection(bbox)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "LineString":
            coast_lines.append(clipped)
        elif clipped.geom_type == "MultiLineString":
            coast_lines.extend(clipped.geoms)
    if not coast_lines:
        return None

    merged = shapely.unary_union([bbox.boundary] + coast_lines)
    polygons = list(shapely.ops.polygonize(merged))
    sea_parts = [p for p in polygons if _on_sea_side(p, coast_lines)]
    if not sea_parts:
        return None

    # Drop tiny sliver polygons — they're usually classification errors where
    # the coastline graph produced small faces along the shore whose nearest
    # segment happens to classify them as "sea" when they're really land.
    # Keep anything within 1 % of the largest sea part's area, floored at
    # 1 000 m² so small bboxes still work.
    max_area = max(p.area for p in sea_parts)
    threshold = max(max_area * 0.01, 1000.0)
    kept = [p for p in sea_parts if p.area >= threshold]
    dropped = len(sea_parts) - len(kept)
    if dropped:
        log.info("sea polygon: dropped %d sliver(s) below %.0f m²",
                 dropped, threshold)
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return shapely.unary_union(kept)
