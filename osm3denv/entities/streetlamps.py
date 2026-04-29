"""Street lamp posts placed along secondary and residential roads.

Lamps are spaced every ``_LAMP_SPACING`` metres along the road centreline,
offset perpendicular to the road on both sides to simulate pavement placement.
The model (``street_lamp.glb``) must be placed manually in
``<cache_dir>/streetfurniture/``.
"""
from __future__ import annotations

import builtins
import logging
import math

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.entities.utils import sample_z
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.render.helpers import nearest_k_idx, tod_intensity

log = logging.getLogger(__name__)

_LAMP_TARGET_H = 7.0   # target rendered height in metres
_LAMP_SPACING  = 50.0  # metres between posts along centreline
_LOD_DIST      = 400.0 # posts disappear beyond this distance

# Spotlight pool
_SPOT_POOL    = 6
_SPOT_COLOR   = (1.0, 0.88, 0.40, 1.0)  # warm sodium-vapour yellow
_SPOT_FOV     = 50.0   # cone full-angle in degrees
_SPOT_RANGE   = 28.0   # metres (lens far plane)
_SPOT_ATTEN   = (1.0, 0.0, 0.014)       # (constant, linear, quadratic)

# highway type → lateral offset from centreline (metres)
# = half road width + ~1 m sidewalk inset
_LAMP_ROADS: dict[str, float] = {
    "secondary":      4.5,
    "secondary_link": 4.5,
    "residential":    3.0,
    "living_street":  3.0,
    "unclassified":   3.0,
}


def _sample_along(east, north, spacing):
    """Yield (e, n, tx, ty) at *spacing*-metre intervals along the polyline."""
    accum  = 0.0
    next_d = spacing * 0.5   # start half-spacing from the first vertex
    for i in range(len(east) - 1):
        e0, n0  = float(east[i]),   float(north[i])
        e1, n1  = float(east[i+1]), float(north[i+1])
        dx, dy  = e1 - e0, n1 - n0
        seg_len = float(np.hypot(dx, dy))
        if seg_len < 1e-3:
            continue
        tx, ty = dx / seg_len, dy / seg_len
        while next_d <= accum + seg_len:
            t = (next_d - accum) / seg_len
            yield e0 + t * dx, n0 + t * dy, tx, ty
            next_d += spacing
        accum += seg_len


def _load_gltf(path):
    import gltf
    from panda3d.core import NodePath
    return NodePath(gltf.load_model(str(path)))


def _natural_height(model_np) -> tuple[float, float]:
    """Return (height_m, base_z_m); auto-converts centimetre exports."""
    bounds = model_np.getTightBounds()
    if not bounds:
        return 1.0, 0.0
    min_z = float(bounds[0].z)
    max_z = float(bounds[1].z)
    h = max(max_z - min_z, 1e-3)
    if h > 200.0:          # heuristic: centimetres
        h    /= 100.0
        min_z /= 100.0
    return h, min_z


class StreetLamps(MapEntity):
    """Lamp posts placed along secondary and residential roads."""

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, terrain, cache_dir) -> None:
        self._osm       = osm
        self._frame     = frame
        self._radius_m  = radius_m
        self._terrain   = terrain
        self._cache_dir = cache_dir
        # (east, north, terrain_z, heading_deg)
        self._positions: list[tuple[float, float, float, float]] = []

    def build(self) -> None:
        td        = self._terrain.data
        heightmap = td.heightmap
        grid      = heightmap.shape[0]
        r         = float(self._radius_m)

        for way in self._osm.filter_ways(lambda t: t.get("highway") in _LAMP_ROADS):
            geom = way.geometry
            if len(geom) < 2:
                continue
            lons = np.fromiter((p[0] for p in geom), np.float64, len(geom))
            lats = np.fromiter((p[1] for p in geom), np.float64, len(geom))
            east, north = self._frame.to_enu(lons, lats)
            if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                continue

            offset = _LAMP_ROADS[way.tags["highway"]]

            for e, n, tx, ty in _sample_along(east, north, _LAMP_SPACING):
                # perpendicular: left = (-ty, tx), right = (ty, -tx)
                px, py = -ty, tx
                for side in (+1.0, -1.0):
                    pe = e + side * px * offset
                    pn = n + side * py * offset
                    if abs(pe) > r or abs(pn) > r:
                        continue
                    z = sample_z(pe, pn, heightmap, grid, r)
                    # Lamp arm points inward toward the road centreline:
                    #   inward direction for side s is (s·ty, -s·tx)
                    heading = math.degrees(math.atan2(side * ty, -side * tx))
                    self._positions.append((pe, pn, z, heading))

        log.info("streetlamps: %d positions", len(self._positions))

    def attach_to(self, parent) -> None:
        # ---- Spotlight pool — always set up regardless of model availability ---
        self._render  = parent
        self._pos_arr = (
            np.array([(e, n, z) for e, n, z, _ in self._positions], dtype=np.float32)
            if self._positions else None
        )
        self._all_idx = (
            np.arange(len(self._positions)) if len(self._positions) <= _SPOT_POOL else None
        )
        self._spot_nps: list = []
        self._setup_spotlights()
        builtins.base.taskMgr.add(self._light_task, "lamp_lights")

        # ---- Model instances (optional — lights work without them) -------------
        from osm3denv.fetch.models import fetch_furniture

        paths = fetch_furniture(self._cache_dir)
        if "street_lamp" not in paths:
            log.warning("street_lamp.glb missing — see log above for download URL")
            return

        try:
            src = _load_gltf(paths["street_lamp"])
        except Exception as exc:
            log.warning("street_lamp model load failed: %s", exc)
            return

        nat_h, base_z = _natural_height(src)
        scale = _LAMP_TARGET_H / nat_h if nat_h > 0.0 else 1.0
        log.info("streetlamps: model h=%.2fm base_z=%.2fm → scale=%.3f",
                 nat_h, base_z, scale)

        # Apply common transforms once so all instances share them.
        src.setPos(0.0, 0.0, -base_z * scale)
        src.setScale(scale)

        from panda3d.core import LODNode
        root = parent.attachNewNode("streetlamps")

        for e, n, z, heading in self._positions:
            lod_np = root.attachNewNode(LODNode("lamp"))
            lod_np.setPos(float(e), float(n), float(z))
            lod    = lod_np.node()

            detail = lod_np.attachNewNode("detail")
            detail.setH(float(heading))
            src.instanceTo(detail)   # geometry shared; one copy in RAM
            lod.addSwitch(_LOD_DIST, 0.0)

        log.info("streetlamps: %d instances attached", len(self._positions))

    def _setup_spotlights(self) -> None:
        import math as _math
        from panda3d.core import LColor, LVecBase3f, LVector3, PTA_LVecBase3f, Spotlight

        # PTA arrays let custom GLSL shaders read live spotlight data each frame.
        self._pta_spot_pos = PTA_LVecBase3f.emptyArray(_SPOT_POOL)
        self._pta_spot_col = PTA_LVecBase3f.emptyArray(_SPOT_POOL)
        for i in range(_SPOT_POOL):
            self._pta_spot_pos[i] = LVecBase3f(0.0, 0.0, -99999.0)
            self._pta_spot_col[i] = LVecBase3f(0.0, 0.0, 0.0)

        cos_cutoff = _math.cos(_math.radians(_SPOT_FOV / 2.0))
        self._render.setShaderInput("u_spot_pos",        self._pta_spot_pos)
        self._render.setShaderInput("u_spot_color",      self._pta_spot_col)
        self._render.setShaderInput("u_spot_cos_cutoff", cos_cutoff)
        self._render.setShaderInput("u_spot_atten",      LVector3(*_SPOT_ATTEN))

        for i in range(_SPOT_POOL):
            spot = Spotlight(f"lamp_spot_{i}")
            spot.setColor(LColor(*_SPOT_COLOR))
            spot.setAttenuation(LVector3(*_SPOT_ATTEN))
            lens = spot.getLens()
            lens.setFov(_SPOT_FOV)
            lens.setNear(1.0)
            lens.setFar(_SPOT_RANGE)
            np_ = self._render.attachNewNode(spot)
            np_.setPos(0.0, 0.0, -99999.0)
            np_.setHpr(0.0, -90.0, 0.0)   # aim straight down
            self._render.setLight(np_)
            self._spot_nps.append(np_)

    def _light_task(self, task):
        if not self._spot_nps or self._pos_arr is None:
            return task.cont

        from panda3d.core import LColor, LVecBase3f

        intensity = tod_intensity(getattr(builtins.base, 'time_of_day', 0.5))

        col = LColor(_SPOT_COLOR[0] * intensity, _SPOT_COLOR[1] * intensity,
                     _SPOT_COLOR[2] * intensity, 1.0)
        for sp in self._spot_nps:
            sp.node().setColor(col)

        if intensity < 0.01:
            for sp in self._spot_nps:
                sp.setPos(0.0, 0.0, -99999.0)
            for i in range(_SPOT_POOL):
                self._pta_spot_pos[i] = LVecBase3f(0.0, 0.0, -99999.0)
                self._pta_spot_col[i] = LVecBase3f(0.0, 0.0, 0.0)
            return task.cont

        cam    = builtins.base.camera.getPos()
        idx    = (self._all_idx if self._all_idx is not None
                  else nearest_k_idx(self._pos_arr[:, :2], float(cam.x), float(cam.y), _SPOT_POOL))

        sc = (_SPOT_COLOR[0] * intensity, _SPOT_COLOR[1] * intensity,
              _SPOT_COLOR[2] * intensity)
        for i, j in enumerate(idx):
            e, n, z, _ = self._positions[int(j)]
            lz = float(z) + _LAMP_TARGET_H * 0.88
            self._spot_nps[i].setPos(float(e), float(n), lz)
            self._spot_nps[i].setHpr(0.0, -90.0, 0.0)
            self._pta_spot_pos[i] = LVecBase3f(float(e), float(n), lz)
            self._pta_spot_col[i] = LVecBase3f(*sc)
        for i in range(len(idx), _SPOT_POOL):
            self._pta_spot_pos[i] = LVecBase3f(0.0, 0.0, -99999.0)
            self._pta_spot_col[i] = LVecBase3f(0.0, 0.0, 0.0)

        return task.cont
