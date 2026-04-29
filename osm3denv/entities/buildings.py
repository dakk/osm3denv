"""Buildings entity — procedural buildings from OSM footprint polygons.

Uses the `procbuilding` library (proceduralbuilding/procbuilding) to generate
geometry from the actual OSM polygon shape.

Two LOD levels per building:
  0 – 80 m    full detail  (walls, windows, door, roof)
  80 – 400 m  reduced      (walls + roof, fewer windows)
  > 400 m     invisible
"""
from __future__ import annotations

import builtins
import logging
import math
import sys
from pathlib import Path
from random import Random

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

# Make procbuilding importable from the bundled proceduralbuilding/ subdirectory.
_PB_DIR = Path(__file__).parent.parent.parent / "proceduralbuilding"
if _PB_DIR.is_dir() and str(_PB_DIR) not in sys.path:
    sys.path.insert(0, str(_PB_DIR))

_MAX_DIM = 80.0
_MIN_DIM =  3.5
_BUDGET  = 5000

_LOD_FULL   =  80.0
_LOD_MEDIUM = 400.0

# Window point-light pool
_BLDG_LIGHT_POOL  = 8
_BLDG_LIGHT_COLOR = (1.0, 0.72, 0.28, 1.0)   # warm incandescent
_BLDG_LIGHT_ATTEN = (1.0, 0.0, 0.025)         # (constant, linear, quadratic)

# Randomised per-building appearance palettes
_WALL_COLORS = [
    (0.92, 0.87, 0.78, 1.0),  # cream
    (0.88, 0.85, 0.80, 1.0),  # warm white
    (0.88, 0.82, 0.68, 1.0),  # light yellow
    (0.90, 0.80, 0.72, 1.0),  # pale peach
    (0.75, 0.75, 0.73, 1.0),  # light grey
    (0.85, 0.82, 0.78, 1.0),  # off-white
    (0.72, 0.38, 0.26, 1.0),  # brick red
    (0.65, 0.63, 0.60, 1.0),  # concrete
]

_ROOF_COLORS = [
    (0.55, 0.22, 0.12, 1.0),  # red clay
    (0.32, 0.32, 0.35, 1.0),  # dark grey
    (0.60, 0.28, 0.18, 1.0),  # terracotta
    (0.40, 0.22, 0.12, 1.0),  # dark brown
    (0.38, 0.36, 0.40, 1.0),  # slate
]


class Buildings(MapEntity):
    """Renders every OSM building=* footprint as a procedural 3-D structure."""

    def __init__(self, osm: OSMData, frame: Frame, radius_m: float,
                 terrain) -> None:
        self._osm      = osm
        self._frame    = frame
        self._radius_m = radius_m
        self._terrain  = terrain
        # (centroid_e, centroid_n, local_verts, floors, bldg_id)
        self._entries: list[tuple] = []

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

        ce = float(east.mean())
        cn = float(north.mean())

        # Build local-frame vertex list, dropping OSM closing duplicate
        raw = list(zip(east.tolist(), north.tolist()))
        if len(raw) > 1 and raw[0] == raw[-1]:
            raw = raw[:-1]
        if len(raw) < 3:
            return None

        local_verts = [(float(e - ce), float(n - cn)) for e, n in raw]
        return (ce, cn, local_verts, floors, bldg_id)

    def attach_to(self, parent) -> None:
        td = self._terrain.data

        # ---- Window point-light pool — always set up regardless of procbuilding --
        self._bldg_light_positions: list[tuple] = []
        for ce, cn, _, floors, _ in self._entries:
            z = self._sample_z(ce, cn, td)
            win_z = float(z) + min(floors, 2) * 3.0 + 1.5
            self._bldg_light_positions.append((float(ce), float(cn), win_z))

        self._bldg_pos_arr = (
            np.array([(e, n) for e, n, _ in self._bldg_light_positions],
                     dtype=np.float32)
            if self._bldg_light_positions else None
        )

        from panda3d.core import LColor, LVector3, PointLight
        self._bldg_pts: list = []
        for i in range(_BLDG_LIGHT_POOL):
            pl = PointLight(f"bwin_{i}")
            pl.setColor(LColor(*_BLDG_LIGHT_COLOR))
            pl.setAttenuation(LVector3(*_BLDG_LIGHT_ATTEN))
            np_ = parent.attachNewNode(pl)
            np_.setPos(0.0, 0.0, -99999.0)
            parent.setLight(np_)
            self._bldg_pts.append(np_)

        builtins.base.taskMgr.add(self._bldg_light_task, "bldg_lights")

        # ---- Building geometry (optional — lights work without procbuilding) ----
        try:
            from procbuilding import PolygonHouse, PolygonHouseParams  # type: ignore[import]
        except ImportError as exc:
            log.warning("procbuilding not available — buildings disabled: %s", exc)
            return

        from panda3d.core import LODNode

        for ce, cn, local_verts, floors, bldg_id in self._entries:
            if len(local_verts) < 3:
                continue
            z = self._sample_z(ce, cn, td)
            rng = Random(bldg_id)

            wall_color = rng.choice(_WALL_COLORS)
            roof_color = rng.choice(_ROOF_COLORS)

            lod_np = parent.attachNewNode(LODNode(f"bld_{bldg_id}"))
            lod_np.setPos(float(ce), float(cn), float(z))
            lod = lod_np.node()

            try:
                full = lod_np.attachNewNode("full")
                PolygonHouse(PolygonHouseParams(
                    verts=local_verts,
                    num_floors=floors,
                    wall_color=wall_color,
                    roof_color=roof_color,
                    windows_per_wall=8,
                )).build(full)
                lod.addSwitch(_LOD_FULL, 0.0)

                medium = lod_np.attachNewNode("medium")
                PolygonHouse(PolygonHouseParams(
                    verts=local_verts,
                    num_floors=floors,
                    wall_color=wall_color,
                    roof_color=roof_color,
                    windows_per_wall=2,
                )).build(medium)
                lod.addSwitch(_LOD_MEDIUM, _LOD_FULL)

            except Exception as exc:
                log.warning("bld %d failed: %s", bldg_id, exc)
                lod_np.removeNode()

    def _bldg_light_task(self, task):
        if not self._bldg_pts or self._bldg_pos_arr is None:
            return task.cont

        tod    = getattr(builtins.base, 'time_of_day', 0.5)
        sin_el = -math.cos(2.0 * math.pi * tod)
        intensity = float(np.clip((-sin_el + 0.05) / 0.15, 0.0, 1.0))

        from panda3d.core import LColor
        col = LColor(_BLDG_LIGHT_COLOR[0] * intensity,
                     _BLDG_LIGHT_COLOR[1] * intensity,
                     _BLDG_LIGHT_COLOR[2] * intensity, 1.0)
        for pl_np in self._bldg_pts:
            pl_np.node().setColor(col)

        if intensity < 0.01:
            for pl_np in self._bldg_pts:
                pl_np.setPos(0.0, 0.0, -99999.0)
            return task.cont

        cam    = builtins.base.camera.getPos()
        ce, cn = float(cam.x), float(cam.y)

        diffs = self._bldg_pos_arr - np.array([ce, cn], dtype=np.float32)
        dists = (diffs * diffs).sum(axis=1)
        n_all = len(self._bldg_pos_arr)
        k     = min(_BLDG_LIGHT_POOL, n_all)
        idx   = np.argpartition(dists, k - 1)[:k] if k < n_all else np.arange(n_all)

        for i, j in enumerate(idx):
            e, n, z = self._bldg_light_positions[int(j)]
            self._bldg_pts[i].setPos(float(e), float(n), float(z))

        return task.cont

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
