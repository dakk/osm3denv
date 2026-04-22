"""Build inland water features (lakes, ponds, rivers) from OSM data.

Lakes / river-bank polygons
---------------------------
Closed ways with natural=water, waterway=riverbank, or landuse=reservoir,
plus multipolygon relations with the same tags.  Each polygon is stored with
a flat *surface_z* equal to the mean terrain height at its perimeter vertices
plus a small upward offset (0.2 m) so it sits clearly above the terrain.

Rivers / streams / canals
--------------------------
Open ways with waterway=river/stream/canal/ditch/drain are stored as
3-D polylines that follow the terrain height (+ 0.3 m offset).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import shapely.geometry as sg
import shapely.ops

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_WATER_AREA_TAGS = {
    ("natural", "water"),
    ("waterway", "riverbank"),
    ("landuse", "reservoir"),
}

_RIVER_LINE_VALUES = {"river", "stream", "canal", "ditch", "drain"}


def _is_water_area(tags: dict) -> bool:
    return any(tags.get(k) == v for k, v in _WATER_AREA_TAGS)


def _is_river_line(tags: dict) -> bool:
    return tags.get("waterway") in _RIVER_LINE_VALUES


@dataclass
class WaterData:
    # Each entry: (shapely Polygon/MultiPolygon in ENU coords, surface z)
    lake_polygons: list = field(default_factory=list)
    # Each entry: (N, 3) float32 — east, north, z (terrain-following + offset)
    rivers: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _grid_coords(x: float, y: float, grid: int, radius_m: float) -> tuple[float, float]:
    col = (x + radius_m) / (2.0 * radius_m) * (grid - 1)
    row = (radius_m - y) / (2.0 * radius_m) * (grid - 1)
    return float(np.clip(row, 0, grid - 1)), float(np.clip(col, 0, grid - 1))


def _bilinear(arr: np.ndarray, row_f: float, col_f: float, grid: int) -> float:
    r0 = min(int(row_f), grid - 2)
    c0 = min(int(col_f), grid - 2)
    fr, fc = row_f - r0, col_f - c0
    return float(
        arr[r0,   c0  ] * (1 - fr) * (1 - fc) +
        arr[r0,   c0+1] * (1 - fr) * fc +
        arr[r0+1, c0  ] * fr       * (1 - fc) +
        arr[r0+1, c0+1] * fr       * fc
    )


def _sample_z(x: float, y: float, heightmap: np.ndarray, grid: int, radius_m: float) -> float:
    row, col = _grid_coords(x, y, grid, radius_m)
    return _bilinear(heightmap, row, col, grid)


def _to_enu(lons, lats, frame: Frame) -> np.ndarray:
    east, north = frame.to_enu(
        np.array(lons, dtype=np.float64),
        np.array(lats, dtype=np.float64),
    )
    return np.stack([east, north], axis=-1).astype(np.float32)


def _ring_enu(ring: list[tuple[float, float]], frame: Frame) -> list[tuple[float, float]]:
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    east, north = frame.to_enu(np.array(lons, dtype=np.float64),
                                np.array(lats, dtype=np.float64))
    return list(zip(east.tolist(), north.tolist()))


def _surface_z(poly, heightmap: np.ndarray, grid: int, radius_m: float,
               offset: float = 0.2) -> float:
    """Mean terrain height at the exterior ring vertices, plus *offset* metres."""
    if poly.geom_type == "MultiPolygon":
        coords = list(poly.geoms[0].exterior.coords)
    else:
        coords = list(poly.exterior.coords)
    heights = [_sample_z(x, y, heightmap, grid, radius_m) for x, y in coords]
    return float(np.mean(heights)) + offset


def _add_polygon(poly, heightmap, grid, radius_m, bbox, out_polygons):
    """Clip poly to bbox, check minimum area, compute surface z, append."""
    try:
        clipped = poly.intersection(bbox)
    except Exception:
        return
    if clipped.is_empty:
        return
    if clipped.geom_type not in ("Polygon", "MultiPolygon"):
        return
    if clipped.area < 10.0:   # skip slivers < 10 m²
        return
    z = _surface_z(clipped, heightmap, grid, radius_m)
    out_polygons.append((clipped, z))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build(osm: OSMData, frame: Frame, radius_m: float,
          heightmap: np.ndarray, h0: float) -> WaterData:
    """Extract water polygons and river polylines from *osm*.

    Parameters
    ----------
    heightmap:
        The (G, G) float32 terrain heightmap from :mod:`osm3denv.mesh.terrain`
        (already zero-centred at the origin, sea vertices clamped).
    h0:
        Absolute SRTM altitude of the scene origin (``TerrainData.origin_alt_m``).
    """
    grid = heightmap.shape[0]
    r = float(radius_m)
    bbox = sg.box(-r, -r, r, r)
    data = WaterData()

    # ------------------------------------------------------------------
    # Closed-way water-area polygons
    # ------------------------------------------------------------------
    for way in osm.filter_ways(_is_water_area):
        geom = way.geometry
        if len(geom) < 3 or geom[0] != geom[-1]:
            continue                          # skip open / degenerate ways
        pts = _to_enu([g[0] for g in geom], [g[1] for g in geom], frame)
        try:
            poly = sg.Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue
        _add_polygon(poly, heightmap, grid, radius_m, bbox, data.lake_polygons)

    # ------------------------------------------------------------------
    # Relation water-area polygons (lakes with islands, large rivers)
    # ------------------------------------------------------------------
    for rel in osm.filter_relations(_is_water_area):
        outer_rings = [ring for role, ring in rel.rings if role in ("outer", "")]
        inner_rings = [ring for role, ring in rel.rings if role == "inner"]
        if not outer_rings:
            continue
        try:
            outer_polys = [sg.Polygon(_ring_enu(rg, frame)) for rg in outer_rings]
            inner_polys = [sg.Polygon(_ring_enu(rg, frame)) for rg in inner_rings]
            poly = shapely.ops.unary_union(outer_polys)
            if inner_polys:
                poly = poly.difference(shapely.ops.unary_union(inner_polys))
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue
        _add_polygon(poly, heightmap, grid, radius_m, bbox, data.lake_polygons)

    # ------------------------------------------------------------------
    # River / stream / canal centre-lines
    # ------------------------------------------------------------------
    for way in osm.filter_ways(_is_river_line):
        geom = way.geometry
        if len(geom) < 2:
            continue
        pts = _to_enu([g[0] for g in geom], [g[1] for g in geom], frame)
        inside = (np.abs(pts[:, 0]) <= r) & (np.abs(pts[:, 1]) <= r)
        segment: list[list[float]] = []
        for i in range(len(pts)):
            if inside[i]:
                x, y = float(pts[i, 0]), float(pts[i, 1])
                z = _sample_z(x, y, heightmap, grid, radius_m) + 0.3
                segment.append([x, y, z])
            elif segment:
                if len(segment) >= 2:
                    data.rivers.append(np.array(segment, dtype=np.float32))
                segment = []
        if len(segment) >= 2:
            data.rivers.append(np.array(segment, dtype=np.float32))

    log.info(
        "water: %d lake/pond polygons, %d river/stream segments",
        len(data.lake_polygons), len(data.rivers),
    )
    return data
