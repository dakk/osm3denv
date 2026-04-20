"""Shared OSM-geometry → Shapely-polygon helpers used across mesh builders."""
from __future__ import annotations

import re

import numpy as np
import shapely
import shapely.geometry as sg
from shapely.geometry.polygon import orient as _orient

from osm3denv.fetch.osm import OSMRelation, OSMWay
from osm3denv.frame import Frame

_NUM_RE = re.compile(r"([-+]?\d*\.?\d+)")


def parse_number(s: str | None) -> float | None:
    """Parse the first number out of an OSM string value (e.g. '12 m', '3.5')."""
    if not s:
        return None
    m = _NUM_RE.search(s)
    return float(m.group(1)) if m else None


def ring_to_enu(ring_ll, frame: Frame) -> np.ndarray:
    lon = np.asarray([p[0] for p in ring_ll], dtype=np.float64)
    lat = np.asarray([p[1] for p in ring_ll], dtype=np.float64)
    east, north = frame.to_enu(lon, lat)
    return np.stack([east, north], axis=-1)


def is_closed(ring: np.ndarray, tol: float = 0.5) -> bool:
    return len(ring) >= 3 and np.linalg.norm(ring[0] - ring[-1]) <= tol


def _take_polygon(geom, min_area: float) -> sg.Polygon | None:
    if geom.is_empty:
        return None
    poly: sg.Polygon | None = None
    if geom.geom_type == "Polygon":
        poly = geom if geom.area >= min_area else None
    elif geom.geom_type == "MultiPolygon":
        best = max(geom.geoms, key=lambda g: g.area)
        poly = best if best.area >= min_area else None
    if poly is None:
        return None
    return _orient(poly, sign=1.0)


def polygon_from_way(way: OSMWay, frame: Frame,
                     *, min_area: float = 1.0) -> sg.Polygon | None:
    ring = ring_to_enu(way.geometry, frame)
    if not is_closed(ring):
        return None
    try:
        poly = sg.Polygon(ring[:-1])
        fixed = shapely.make_valid(poly.buffer(0))
    except Exception:  # noqa: BLE001
        return None
    return _take_polygon(fixed, min_area)


def polygons_from_relation(rel: OSMRelation, frame: Frame,
                           *, min_area: float = 1.0) -> list[sg.Polygon]:
    """Build zero or more polygons from a multipolygon relation.

    Each outer ring becomes its own polygon; any inner ring fully contained
    inside an outer becomes a hole.
    """
    outers = [r for (role, r) in rel.rings if role == "outer"]
    inners = [r for (role, r) in rel.rings if role == "inner"]
    results: list[sg.Polygon] = []
    for outer_ll in outers:
        outer = ring_to_enu(outer_ll, frame)
        if not is_closed(outer):
            continue
        try:
            base = sg.Polygon(outer[:-1])
            holes = []
            for inner_ll in inners:
                inner = ring_to_enu(inner_ll, frame)
                if is_closed(inner):
                    h = sg.Polygon(inner[:-1])
                    if base.contains(h):
                        holes.append(list(h.exterior.coords)[:-1])
            base = sg.Polygon(list(base.exterior.coords)[:-1], holes)
            fixed = shapely.make_valid(base.buffer(0))
        except Exception:  # noqa: BLE001
            continue
        poly = _take_polygon(fixed, min_area)
        if poly is not None:
            results.append(poly)
    return results


def polygon_from_relation(rel: OSMRelation, frame: Frame,
                          *, min_area: float = 1.0) -> sg.Polygon | None:
    """Back-compat: return the largest polygon extracted from the relation."""
    polys = polygons_from_relation(rel, frame, min_area=min_area)
    if not polys:
        return None
    return max(polys, key=lambda p: p.area)
