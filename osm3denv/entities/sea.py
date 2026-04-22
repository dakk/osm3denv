"""Sea entity — ocean plane with Gerstner wave shader."""
from __future__ import annotations

import logging

import numpy as np
import shapely
import shapely.geometry as sg
import shapely.ops

from osm3denv.entity import MapEntity
from osm3denv.entities.utils import triangulate_flat_poly
from osm3denv.fetch.osm import OSMData, OSMWay
from osm3denv.frame import Frame

log = logging.getLogger(__name__)


def _coastline_to_enu(way: OSMWay, frame: Frame) -> sg.LineString | None:
    if len(way.geometry) < 2:
        return None
    lon = np.fromiter((p[0] for p in way.geometry), dtype=np.float64, count=len(way.geometry))
    lat = np.fromiter((p[1] for p in way.geometry), dtype=np.float64, count=len(way.geometry))
    east, north = frame.to_enu(lon, lat)
    line = sg.LineString(list(zip(east, north)))
    return line if line.length > 0.5 else None


def _on_sea_side(poly: sg.Polygon, coast_lines: list[sg.LineString]) -> bool:
    p = poly.representative_point()
    px, py = p.x, p.y
    best_d, best_cross = float("inf"), 0.0
    for line in coast_lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            ax, ay = coords[i];  bx, by = coords[i + 1]
            dx, dy = bx - ax, by - ay
            seg_len2 = dx*dx + dy*dy
            if seg_len2 <= 1e-12:
                continue
            t = max(0.0, min(1.0, ((px - ax)*dx + (py - ay)*dy) / seg_len2))
            nx, ny = ax + t*dx, ay + t*dy
            d = float(np.hypot(px - nx, py - ny))
            if d < best_d:
                best_d = d
                best_cross = dx*(py - ay) - dy*(px - ax)
    return best_cross < 0.0


class Sea(MapEntity):
    """Ocean plane entity.

    Two-phase build required by the Terrain dependency::

        sea = Sea(osm, frame, radius_m)
        sea.build()               # extracts sea polygon from OSM coastlines
        terrain.build()           # terrain needs sea.polygon to clamp underwater verts
        sea.finalize(terrain)     # sea needs terrain.data.origin_alt_m for z-placement
    """

    SHADER = "sea"

    def __init__(self, osm: OSMData, frame: Frame, radius_m: float) -> None:
        self._osm = osm
        self._frame = frame
        self._radius_m = radius_m
        self._polygon = None
        self._sea_z: float = 0.0
        self._verts: np.ndarray | None = None
        self._norms: np.ndarray | None = None

    # ------------------------------------------------------------------
    def build(self) -> None:
        """Phase 1 — extract sea polygon from OSM coastlines."""
        coast_ways = self._osm.filter_ways(lambda t: t.get("natural") == "coastline")
        if not coast_ways:
            log.info("sea: no coastlines in bbox")
            return

        r = self._radius_m
        bbox = sg.box(-r, -r, r, r)
        coast_lines: list[sg.LineString] = []
        for w in coast_ways:
            line = _coastline_to_enu(w, self._frame)
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
            return

        merged = shapely.unary_union([bbox.boundary] + coast_lines)
        polygons = list(shapely.ops.polygonize(merged))
        sea_parts = [p for p in polygons if _on_sea_side(p, coast_lines)]
        if not sea_parts:
            return

        max_area = max(p.area for p in sea_parts)
        threshold = max(max_area * 0.01, 1000.0)
        kept = [p for p in sea_parts if p.area >= threshold]
        dropped = len(sea_parts) - len(kept)
        if dropped:
            log.info("sea: dropped %d sliver(s) below %.0f m²", dropped, threshold)
        if not kept:
            return

        self._polygon = kept[0] if len(kept) == 1 else shapely.unary_union(kept)
        log.info("sea polygon: area=%.0f m² (%s)", self._polygon.area, self._polygon.geom_type)

    def finalize(self, terrain) -> None:
        """Phase 2 — build geometry. Call after terrain.build()."""
        if self._polygon is None:
            return
        r = self._radius_m
        e = r * 20.0
        self._sea_z = -terrain.data.origin_alt_m - 0.3
        z = self._sea_z

        outer = [
            [(-e, e), (r, r), (e, e)],   [(-e, e), (-r, r), (r, r)],
            [(-e,-e), (e,-e), (r,-r)],   [(-e,-e), (r,-r), (-r,-r)],
            [(-e,-e), (-r, r), (-e, e)], [(-e,-e), (-r,-r), (-r, r)],
            [(e, -e), (e, e), (r, r)],   [(e, -e), (r, r), (r,-r)],
        ]
        clipped = self._polygon.intersection(sg.box(-r, -r, r, r))
        inner = [] if clipped.is_empty else triangulate_flat_poly(clipped, max(30.0, r / 30.0))

        all_tris = outer + inner
        self._verts = np.array([[x, y, z] for tri in all_tris for (x, y) in tri],
                                dtype=np.float32)
        self._norms = np.zeros_like(self._verts)
        self._norms[:, 2] = 1.0

    @property
    def polygon(self):
        return self._polygon

    @property
    def sea_z(self) -> float:
        return self._sea_z

    # ------------------------------------------------------------------
    def attach_to(self, parent) -> None:
        if self._verts is None:
            return
        from osm3denv.render.helpers import attach_mesh, load_shader
        np_ = attach_mesh(parent, "sea", self._verts, self._norms, depth_offset=1)
        np_.setTwoSided(True)
        shader = load_shader(self.SHADER)
        if shader:
            np_.setShader(shader)
