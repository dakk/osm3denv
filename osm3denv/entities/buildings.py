"""Buildings entity — procedural residential houses from OSM building footprints.

Three LOD levels per building:
  0–80 m   full detail  (walls, roof, windows, door, balcony, chimney, AC)
  80–300 m medium       (walls + roof only)
  300–700 m simple      (single textured box)
  > 700 m  invisible

Shared textures are pre-loaded once in HouseBuilder so RAM scales with
material types, not building count.
"""
from __future__ import annotations

import logging

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_MAX_DIM = 80.0   # skip buildings larger than this (airports, stadiums)
_MIN_DIM =  3.5
_BUDGET  = 5000

_LOD_FULL   =  80.0
_LOD_MEDIUM = 300.0
_LOD_SIMPLE = 700.0


class Buildings(MapEntity):
    """Renders OSM residential buildings as procedural 3-D houses with LOD."""

    def __init__(self, osm: OSMData, frame: Frame, radius_m: float,
                 terrain, tex_paths: dict | None = None) -> None:
        self._osm       = osm
        self._frame     = frame
        self._radius_m  = radius_m
        self._terrain   = terrain
        self._tex_paths = tex_paths or {}
        self._entries: list[tuple] = []  # (east, north, width, depth, floors, bldg_id)

    def build(self) -> None:
        seen: set[int] = set()

        for way in self._osm.filter_ways(lambda t: bool(t.get("building"))):
            if len(self._entries) >= _BUDGET:
                break
            if way.id in seen:
                continue
            entry = self._process_geometry(way.geometry, way.tags, way.id)
            if entry:
                self._entries.append(entry)
                seen.add(way.id)

        for rel in self._osm.filter_relations(lambda t: bool(t.get("building"))):
            if len(self._entries) >= _BUDGET:
                break
            if rel.id in seen:
                continue
            for role, ring in rel.rings:
                if role not in ("outer", "") or len(ring) < 4:
                    continue
                entry = self._process_geometry(ring, rel.tags, rel.id)
                if entry:
                    self._entries.append(entry)
                    seen.add(rel.id)
                    break

        log.info("buildings: %d structures queued", len(self._entries))

    def _process_geometry(self, pts, tags: dict, bldg_id: int) -> tuple | None:
        if len(pts) < 4:
            return None
        r    = self._radius_m
        lons = np.fromiter((p[0] for p in pts), np.float64, len(pts))
        lats = np.fromiter((p[1] for p in pts), np.float64, len(pts))
        east, north = self._frame.to_enu(lons, lats)

        if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
            return None

        width = float(east.max() - east.min())
        depth = float(north.max() - north.min())
        if max(width, depth) > _MAX_DIM or min(width, depth) < _MIN_DIM:
            return None

        floors = 2
        if "building:levels" in tags:
            try:
                floors = max(1, min(12, int(float(tags["building:levels"]))))
            except ValueError:
                pass
        elif "height" in tags:
            try:
                floors = max(1, min(12, round(float(tags["height"].split()[0]) / 3.0)))
            except ValueError:
                pass

        return (float(east.mean()), float(north.mean()), width, depth, floors, bldg_id)

    def attach_to(self, parent) -> None:
        from panda3d.core import LODNode
        from osm3denv.render.helpers import load_shader
        from osm3denv.render.procedural.house import HouseBuilder

        shader  = load_shader("building")
        builder = HouseBuilder(self._tex_paths, shader=shader)
        td      = self._terrain.data

        for cx, cy, width, depth, floors, bldg_id in self._entries:
            z = self._sample_z(cx, cy, td)

            lod_np = parent.attachNewNode(LODNode(f"bld_{bldg_id}"))
            lod_np.setPos(float(cx), float(cy), float(z))
            lod    = lod_np.node()

            full   = lod_np.attachNewNode("full")
            builder.build_full(bldg_id, width, depth, floors, full)
            lod.addSwitch(_LOD_FULL, 0.0)

            medium = lod_np.attachNewNode("medium")
            builder.build_medium(bldg_id, width, depth, floors, medium)
            lod.addSwitch(_LOD_MEDIUM, _LOD_FULL)

            simple = lod_np.attachNewNode("simple")
            builder.build_simple(bldg_id, width, depth, floors, simple)
            lod.addSwitch(_LOD_SIMPLE, _LOD_MEDIUM)

    def _sample_z(self, east: float, north: float, td) -> float:
        g     = td.heightmap.shape[0]
        r     = float(td.radius_m)
        scale = (g - 1) / (2.0 * r)
        col   = float(np.clip((east  + r) * scale, 0, g - 1))
        row   = float(np.clip((r - north) * scale, 0, g - 1))
        r0    = min(int(row), g - 2)
        c0    = min(int(col), g - 2)
        fr    = row - r0; fc = col - c0
        return float(
            td.heightmap[r0,   c0  ] * (1-fr)*(1-fc) +
            td.heightmap[r0,   c0+1] * (1-fr)*fc     +
            td.heightmap[r0+1, c0  ] * fr    *(1-fc) +
            td.heightmap[r0+1, c0+1] * fr    *fc
        )
