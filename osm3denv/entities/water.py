"""Water entity — lakes, ponds, reservoirs and rivers rendered with the lake shader."""
from __future__ import annotations

import logging

import numpy as np
import shapely.geometry as sg
import shapely.ops

from osm3denv.entity import MapEntity
from osm3denv.entities.utils import sample_z, triangulate_flat_poly
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_WATER_AREA_TAGS = {
    ("natural", "water"),
    ("waterway", "riverbank"),
    ("landuse", "reservoir"),
}
_RIVER_LINE_VALUES = {"river", "stream", "canal", "ditch", "drain"}
_RIVER_HALF_WIDTH: dict[str, float] = {
    "river": 12.0, "canal": 8.0, "stream": 3.0, "ditch": 1.5, "drain": 1.5,
}


def _is_water_area(tags: dict) -> bool:
    return any(tags.get(k) == v for k, v in _WATER_AREA_TAGS)


def _is_river_line(tags: dict) -> bool:
    return tags.get("waterway") in _RIVER_LINE_VALUES


def _river_half_width(tags: dict) -> float:
    raw = tags.get("width")
    if raw is not None:
        try:
            return float(raw) / 2.0
        except ValueError:
            pass
    return _RIVER_HALF_WIDTH.get(tags.get("waterway", "stream"), 3.0)


def _to_enu(lons, lats, frame: Frame) -> np.ndarray:
    east, north = frame.to_enu(np.array(lons, dtype=np.float64),
                                np.array(lats, dtype=np.float64))
    return np.stack([east, north], axis=-1).astype(np.float32)


def _ring_enu(ring, frame: Frame) -> list[tuple[float, float]]:
    lons = [p[0] for p in ring];  lats = [p[1] for p in ring]
    east, north = frame.to_enu(np.array(lons, dtype=np.float64),
                                np.array(lats, dtype=np.float64))
    return list(zip(east.tolist(), north.tolist()))


def _surface_z(poly, heightmap, grid, radius_m, offset=0.2) -> float:
    coords = (list(poly.geoms[0].exterior.coords)
              if poly.geom_type == "MultiPolygon"
              else list(poly.exterior.coords))
    return float(np.mean([sample_z(x, y, heightmap, grid, radius_m)
                          for x, y in coords])) + offset


class Water(MapEntity):
    """Inland water bodies (lakes, ponds, rivers) using the lake shader."""

    SHADER = "lake"

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, terrain) -> None:
        self._osm = osm
        self._frame = frame
        self._radius_m = radius_m
        self._terrain = terrain
        self._verts: np.ndarray | None = None
        self._norms: np.ndarray | None = None

    def build(self) -> None:
        td = self._terrain.data
        heightmap, h0 = td.heightmap, td.origin_alt_m
        grid = heightmap.shape[0]
        r = float(self._radius_m)
        bbox = sg.box(-r, -r, r, r)
        lake_polygons: list[tuple] = []

        for way in self._osm.filter_ways(_is_water_area):
            geom = way.geometry
            if len(geom) < 3 or geom[0] != geom[-1]:
                continue
            pts = _to_enu([g[0] for g in geom], [g[1] for g in geom], self._frame)
            try:
                poly = sg.Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                continue
            self._add_polygon(poly, heightmap, grid, r, bbox, lake_polygons)

        for rel in self._osm.filter_relations(_is_water_area):
            outer = [ring for role, ring in rel.rings if role in ("outer", "")]
            inner = [ring for role, ring in rel.rings if role == "inner"]
            if not outer:
                continue
            try:
                outer_polys = [sg.Polygon(_ring_enu(rg, self._frame)) for rg in outer]
                inner_polys = [sg.Polygon(_ring_enu(rg, self._frame)) for rg in inner]
                poly = shapely.ops.unary_union(outer_polys)
                if inner_polys:
                    poly = poly.difference(shapely.ops.unary_union(inner_polys))
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                continue
            self._add_polygon(poly, heightmap, grid, r, bbox, lake_polygons)

        n_rivers = 0
        for way in self._osm.filter_ways(_is_river_line):
            geom = way.geometry
            if len(geom) < 2:
                continue
            pts = _to_enu([g[0] for g in geom], [g[1] for g in geom], self._frame)
            half_w = _river_half_width(way.tags)
            try:
                poly = sg.LineString(pts).buffer(half_w, cap_style=2)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                continue
            inside = (np.abs(pts[:, 0]) <= r) & (np.abs(pts[:, 1]) <= r)
            centre = pts[inside]
            if len(centre) == 0:
                continue
            z = float(np.mean([sample_z(float(p[0]), float(p[1]), heightmap, grid, r)
                                for p in centre])) + 0.15
            try:
                clipped = poly.intersection(bbox)
            except Exception:
                continue
            if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            if clipped.area < 1.0:
                continue
            lake_polygons.append((clipped, z))
            n_rivers += 1

        log.info("water: %d lake/pond polygons, %d river/stream polygons",
                 len(lake_polygons) - n_rivers, n_rivers)

        if not lake_polygons:
            return

        all_tris: list[list[tuple]] = []
        for poly, surface_z in lake_polygons:
            max_seg = max(10.0, poly.area ** 0.5 / 20.0)
            for tri_coords in triangulate_flat_poly(poly, max_seg):
                all_tris.append([(x, y, surface_z) for (x, y) in tri_coords])

        if not all_tris:
            return

        self._verts = np.array(all_tris, dtype=np.float32).reshape(-1, 3)
        self._norms = np.zeros_like(self._verts)
        self._norms[:, 2] = 1.0

    @staticmethod
    def _add_polygon(poly, heightmap, grid, radius_m, bbox, out):
        try:
            clipped = poly.intersection(bbox)
        except Exception:
            return
        if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
            return
        if clipped.area < 10.0:
            return
        out.append((clipped, _surface_z(clipped, heightmap, grid, radius_m)))

    def attach_to(self, parent) -> None:
        if self._verts is None:
            return
        from osm3denv.render.helpers import attach_mesh, load_shader
        np_ = attach_mesh(parent, "water", self._verts, self._norms, depth_offset=2)
        np_.setTwoSided(True)
        shader = load_shader(self.SHADER)
        if shader:
            np_.setShader(shader)
