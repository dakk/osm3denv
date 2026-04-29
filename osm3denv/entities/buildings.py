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
from collections import deque
from pathlib import Path
from random import Random

import numpy as np

from osm3denv.entities.utils import sample_z
from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.render.helpers import nearest_k_idx, tod_intensity

log = logging.getLogger(__name__)

# Make procbuilding importable from the bundled proceduralbuilding/ subdirectory.
_PB_DIR = Path(__file__).parent.parent.parent / "proceduralbuilding"
if _PB_DIR.is_dir() and str(_PB_DIR) not in sys.path:
    sys.path.insert(0, str(_PB_DIR))

_MAX_DIM = 80.0
_MIN_DIM =  3.5
_BUDGET  = 5000

_LOD_MEDIUM = 400.0

# Streaming — only cells within this radius are in the scene graph
_BLDG_CELL_SIZE     = 200.0   # metres per cell
_BLDG_STREAM_RADIUS = 600.0   # load cells within this distance
_BLDG_UNLOAD_RADIUS = 900.0   # hysteresis: unload beyond this
_BLDG_PER_FRAME     = 8       # buildings built per frame (procbuilding is slow)
_BLDG_CELL_CAP      = 40      # max buildings per cell

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
            added = False
            for role, ring in rel.rings:
                if role not in ("outer", "") or len(ring) < 4:
                    continue
                entry = self._process_geometry(ring, rel.tags, rel.id)
                if entry:
                    self._entries.append(entry)
                    added = True
            if added:
                seen.add(rel.id)

        # Index entries into spatial cells for streaming
        self._bldg_by_cell: dict[tuple, list] = {}
        for entry in self._entries:
            ce, cn = entry[0], entry[1]
            ci = int(ce // _BLDG_CELL_SIZE)
            cj = int(cn // _BLDG_CELL_SIZE)
            bucket = self._bldg_by_cell.setdefault((ci, cj), [])
            if len(bucket) < _BLDG_CELL_CAP:
                bucket.append(entry)

        log.info("buildings: %d structures across %d cells",
                 len(self._entries), len(self._bldg_by_cell))

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
            z = sample_z(ce, cn, td.heightmap, td.heightmap.shape[0], td.radius_m)
            win_z = float(z) + min(floors, 2) * 3.0 + 1.5
            self._bldg_light_positions.append((float(ce), float(cn), win_z))

        self._bldg_pos_arr = (
            np.array([(e, n) for e, n, _ in self._bldg_light_positions],
                     dtype=np.float32)
            if self._bldg_light_positions else None
        )
        self._bldg_all_idx = (
            np.arange(len(self._bldg_light_positions))
            if len(self._bldg_light_positions) <= _BLDG_LIGHT_POOL else None
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

        # ---- Streaming geometry (lights work immediately, geometry streams in) ----
        try:
            from procbuilding import PolygonHouse, PolygonHouseParams  # type: ignore[import]
        except ImportError as exc:
            log.warning("procbuilding not available — buildings disabled: %s", exc)
            return

        self._bldg_root      = parent.attachNewNode("buildings")
        self._PolygonHouse   = PolygonHouse
        self._PolygonHouseParams = PolygonHouseParams
        self._active_cells:  dict[tuple, object] = {}
        self._pending_cells: deque = deque()
        self._cur_cell_key:  tuple | None = None
        self._cur_cell_np    = None
        self._cur_cell_bldgs: deque = deque()
        builtins.base.taskMgr.add(self._bldg_stream_task, "bldg_stream")

    def _bldg_stream_task(self, task):
        from panda3d.core import LODNode

        pos   = builtins.base.camera.getPos()
        cam_e = float(pos.x)
        cam_n = float(pos.y)

        # --- Determine which cells are needed ---
        r_cells = int(_BLDG_STREAM_RADIUS / _BLDG_CELL_SIZE) + 1
        ci_cam  = int(cam_e // _BLDG_CELL_SIZE)
        cj_cam  = int(cam_n // _BLDG_CELL_SIZE)
        needed: set[tuple] = set()
        for dci in range(-r_cells, r_cells + 1):
            for dcj in range(-r_cells, r_cells + 1):
                ci, cj = ci_cam + dci, cj_cam + dcj
                if (ci, cj) not in self._bldg_by_cell:
                    continue
                cx = (ci + 0.5) * _BLDG_CELL_SIZE
                cn = (cj + 0.5) * _BLDG_CELL_SIZE
                if math.sqrt((cx - cam_e) ** 2 + (cn - cam_n) ** 2) <= _BLDG_STREAM_RADIUS:
                    needed.add((ci, cj))

        # --- Queue new cells ---
        queued = set(self._pending_cells)
        for key in needed - set(self._active_cells) - queued:
            self._pending_cells.append(key)

        # --- Unload distant cells ---
        for key in list(self._active_cells):
            ci, cj = key
            cx = (ci + 0.5) * _BLDG_CELL_SIZE
            cn = (cj + 0.5) * _BLDG_CELL_SIZE
            if math.sqrt((cx - cam_e) ** 2 + (cn - cam_n) ** 2) > _BLDG_UNLOAD_RADIUS:
                self._active_cells.pop(key).removeNode()
                if self._cur_cell_key == key:
                    self._cur_cell_bldgs.clear()
                    self._cur_cell_key = None

        # --- Build up to N buildings this frame ---
        for _ in range(_BLDG_PER_FRAME):
            # Advance to next pending cell if current is exhausted
            if not self._cur_cell_bldgs:
                if not self._pending_cells:
                    break
                key = self._pending_cells.popleft()
                if key in self._active_cells:
                    continue
                cell_np = self._bldg_root.attachNewNode(f"bc_{key[0]}_{key[1]}")
                self._active_cells[key] = cell_np
                self._cur_cell_key  = key
                self._cur_cell_np   = cell_np
                self._cur_cell_bldgs = deque(self._bldg_by_cell.get(key, []))

            if not self._cur_cell_bldgs:
                break

            ce, cn, local_verts, floors, bldg_id = self._cur_cell_bldgs.popleft()
            if len(local_verts) < 3:
                continue
            td  = self._terrain.data
            z   = sample_z(ce, cn, td.heightmap, td.heightmap.shape[0], td.radius_m)
            rng = Random(bldg_id)

            lod_np = self._cur_cell_np.attachNewNode(LODNode(f"b_{bldg_id}"))
            lod_np.setPos(float(ce), float(cn), float(z))
            try:
                full = lod_np.attachNewNode("f")
                self._PolygonHouse(self._PolygonHouseParams(
                    verts=local_verts,
                    num_floors=floors,
                    wall_color=rng.choice(_WALL_COLORS),
                    roof_color=rng.choice(_ROOF_COLORS),
                    windows_per_wall=8,
                )).build(full)
                lod_np.node().addSwitch(_LOD_MEDIUM, 0.0)
            except Exception as exc:
                log.debug("bld %d failed: %s", bldg_id, exc)
                lod_np.removeNode()

        return task.cont

    def _bldg_light_task(self, task):
        if not self._bldg_pts or self._bldg_pos_arr is None:
            return task.cont

        from panda3d.core import LColor

        intensity = tod_intensity(getattr(builtins.base, 'time_of_day', 0.5))

        col = LColor(_BLDG_LIGHT_COLOR[0] * intensity,
                     _BLDG_LIGHT_COLOR[1] * intensity,
                     _BLDG_LIGHT_COLOR[2] * intensity, 1.0)
        for pl_np in self._bldg_pts:
            pl_np.node().setColor(col)

        if intensity < 0.01:
            for pl_np in self._bldg_pts:
                pl_np.setPos(0.0, 0.0, -99999.0)
            return task.cont

        cam = builtins.base.camera.getPos()
        idx = (self._bldg_all_idx if self._bldg_all_idx is not None
               else nearest_k_idx(self._bldg_pos_arr, float(cam.x), float(cam.y),
                                   _BLDG_LIGHT_POOL))

        for i, j in enumerate(idx):
            e, n, z = self._bldg_light_positions[int(j)]
            self._bldg_pts[i].setPos(float(e), float(n), float(z))

        return task.cont

