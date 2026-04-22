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

import numpy as np
import shapely.geometry as sg
import shapely.ops

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.layer import RenderLayer
from osm3denv.mesh.utils import sample_z, triangulate_flat_poly

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


def _to_enu(lons, lats, frame: Frame) -> np.ndarray:
    east, north = frame.to_enu(
        np.array(lons, dtype=np.float64),
        np.array(lats, dtype=np.float64),
    )
    return np.stack([east, north], axis=-1).astype(np.float32)


def _ring_enu(ring, frame: Frame) -> list[tuple[float, float]]:
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    east, north = frame.to_enu(np.array(lons, dtype=np.float64),
                                np.array(lats, dtype=np.float64))
    return list(zip(east.tolist(), north.tolist()))


def _surface_z(poly, heightmap: np.ndarray, grid: int, radius_m: float,
               offset: float = 0.2) -> float:
    """Mean terrain height at exterior ring vertices plus *offset* metres."""
    if poly.geom_type == "MultiPolygon":
        coords = list(poly.geoms[0].exterior.coords)
    else:
        coords = list(poly.exterior.coords)
    heights = [sample_z(x, y, heightmap, grid, radius_m) for x, y in coords]
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
    if clipped.area < 10.0:
        return
    z = _surface_z(clipped, heightmap, grid, radius_m)
    out_polygons.append((clipped, z))


def build(osm: OSMData, frame: Frame, radius_m: float,
          heightmap: np.ndarray, h0: float) -> list[RenderLayer]:
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
    lake_polygons: list[tuple] = []

    # ------------------------------------------------------------------
    # Closed-way water-area polygons
    # ------------------------------------------------------------------
    for way in osm.filter_ways(_is_water_area):
        geom = way.geometry
        if len(geom) < 3 or geom[0] != geom[-1]:
            continue
        pts = _to_enu([g[0] for g in geom], [g[1] for g in geom], frame)
        try:
            poly = sg.Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue
        _add_polygon(poly, heightmap, grid, radius_m, bbox, lake_polygons)

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
        _add_polygon(poly, heightmap, grid, radius_m, bbox, lake_polygons)

    # ------------------------------------------------------------------
    # River / stream / canal centre-lines
    # ------------------------------------------------------------------
    river_polylines: list[np.ndarray] = []
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
                z = sample_z(x, y, heightmap, grid, radius_m) + 0.3
                segment.append([x, y, z])
            elif segment:
                if len(segment) >= 2:
                    river_polylines.append(np.array(segment, dtype=np.float32))
                segment = []
        if len(segment) >= 2:
            river_polylines.append(np.array(segment, dtype=np.float32))

    log.info("water: %d lake/pond polygons, %d river/stream segments",
             len(lake_polygons), len(river_polylines))

    layers: list[RenderLayer] = []

    if lake_polygons:
        all_tris: list[list[tuple]] = []
        for poly, surface_z in lake_polygons:
            max_seg = max(30.0, poly.area ** 0.5 / 20.0)
            for tri_coords in triangulate_flat_poly(poly, max_seg):
                all_tris.append([(x, y, surface_z) for (x, y) in tri_coords])
        if all_tris:
            verts = np.array(all_tris, dtype=np.float32).reshape(-1, 3)
            norms = np.zeros_like(verts)
            norms[:, 2] = 1.0
            layers.append(RenderLayer(
                name="water_lakes",
                vertices=verts,
                normals=norms,
                color=(0.15, 0.40, 0.65, 1.0),
                depth_offset=2,
                two_sided=True,
                shader_name="lake",
            ))

    if river_polylines:
        layers.append(RenderLayer(
            name="water_rivers",
            polylines=river_polylines,
            line_thickness=2.0,
            color=(0.15, 0.45, 0.75, 1.0),
            lit=False,
        ))

    return layers
