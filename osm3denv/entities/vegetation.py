"""Vegetation entity — camera-driven streaming trees and ground cover.

OSM polygon areas are indexed per 200 m cell at startup.  As the camera
moves a per-frame task loads cells entering a stream radius and unloads
cells leaving an unload radius.  Only the nearby ring of cells is ever in
the scene graph, so density is limited only by the per-cell cap rather
than a global budget.

Models by evolveduk (CC BY 4.0): https://sketchfab.com/evolveduk/models
"""
from __future__ import annotations

import builtins
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streaming parameters
# ---------------------------------------------------------------------------
_CELL_SIZE              = 200.0   # metres per spatial cell
_STREAM_RADIUS          = 1000.0  # load cells whose centre is within this range
_UNLOAD_RADIUS          = 1400.0  # unload cells whose centre exceeds this range
_MAX_CELL_TREES         = 600     # hard cap on tree/shrub plants per cell
_MAX_LOADS_PER_FRAME    = 4       # max cells to build per frame (avoids stutter)
_GROUNDCOVER_SPACING    = 1.2     # metres between grass_claster / daisy instances
_MAX_CELL_GROUNDCOVER   = 500     # hard cap on ground-cover instances per cell


# ---------------------------------------------------------------------------
# VegType
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VegType:
    h_min:   float
    h_max:   float
    spacing: float
    jitter:  float


_VEG_TYPES: dict[str, VegType] = {
    "park":          VegType(h_min=7.0,  h_max=14.0, spacing=20.0, jitter=0.50),
    "orchard":       VegType(h_min=3.0,  h_max=6.0,  spacing=6.0,  jitter=0.08),
    "scrub":         VegType(h_min=0.5,  h_max=2.0,  spacing=1.5,  jitter=0.50),
    "heath":         VegType(h_min=0.2,  h_max=0.8,  spacing=0.8,  jitter=0.60),
    "cemetery":      VegType(h_min=8.0,  h_max=16.0, spacing=15.0, jitter=0.30),
    "garden":        VegType(h_min=3.0,  h_max=7.0,  spacing=6.0,  jitter=0.50),
    "village_green": VegType(h_min=8.0,  h_max=14.0, spacing=20.0, jitter=0.40),
    "allotments":    VegType(h_min=2.0,  h_max=4.0,  spacing=6.0,  jitter=0.30),
    "tree":          VegType(h_min=6.0,  h_max=12.0, spacing=0.0,  jitter=0.0),
    "forest":        VegType(h_min=10.0, h_max=18.0, spacing=12.0, jitter=0.40),
    "residential":   VegType(h_min=4.0,  h_max=12.0, spacing=15.0, jitter=0.50),
}


# ---------------------------------------------------------------------------
# OSM classification
# ---------------------------------------------------------------------------

def _classify(tags: dict) -> str | None:
    landuse = tags.get("landuse")
    natural = tags.get("natural")
    leisure = tags.get("leisure")
    amenity = tags.get("amenity")
    if landuse == "forest" or natural == "wood":         return "forest"
    if leisure == "park":                                return "park"
    if landuse == "orchard":                             return "orchard"
    if natural == "scrub":                               return "scrub"
    if natural == "heath":                               return "heath"
    if landuse == "cemetery" or amenity == "grave_yard": return "cemetery"
    if leisure == "garden":                              return "garden"
    if landuse == "village_green":                       return "village_green"
    if landuse == "allotments":                          return "allotments"
    if landuse == "residential":                         return "residential"
    return None


# ---------------------------------------------------------------------------
# Terrain sampling
# ---------------------------------------------------------------------------

def _sample_z_triangle_vec(e_arr, n_arr, heightmap, grid, radius_m):
    scale = (grid - 1) / (2.0 * radius_m)
    col_f = np.clip((np.asarray(e_arr, np.float64) + radius_m) * scale, 0.0, grid - 1)
    row_f = np.clip((radius_m - np.asarray(n_arr, np.float64)) * scale, 0.0, grid - 1)
    r0 = np.minimum(row_f.astype(np.int32), grid - 2)
    c0 = np.minimum(col_f.astype(np.int32), grid - 2)
    fr = (row_f - r0).astype(np.float32)
    fc = (col_f - c0).astype(np.float32)
    z_nw = heightmap[r0,     c0    ].astype(np.float32)
    z_ne = heightmap[r0,     c0 + 1].astype(np.float32)
    z_sw = heightmap[r0 + 1, c0    ].astype(np.float32)
    z_se = heightmap[r0 + 1, c0 + 1].astype(np.float32)
    lower_right = (fr + fc) >= 1.0
    return np.where(
        lower_right,
        (fc + fr - 1.0) * z_se + (1.0 - fr) * z_ne + (1.0 - fc) * z_sw,
        fc * z_ne + (1.0 - fr - fc) * z_nw + fr * z_sw,
    )


# ---------------------------------------------------------------------------
# Point-in-polygon test
# ---------------------------------------------------------------------------

def _points_in_polygon(pts_e, pts_n, poly_e, poly_n):
    inside = np.zeros(len(pts_e), dtype=bool)
    j = len(poly_e) - 1
    for i in range(len(poly_e)):
        xi, yi = poly_e[i], poly_n[i]
        xj, yj = poly_e[j], poly_n[j]
        crosses = (yi > pts_n) != (yj > pts_n)
        x_int   = (xj - xi) * (pts_n - yi) / ((yj - yi) + 1e-12) + xi
        inside ^= crosses & (pts_e < x_int)
        j = i
    return inside


# ---------------------------------------------------------------------------
# Cell-aware scatter — only generates points within the cell bbox
# ---------------------------------------------------------------------------

def _scatter_in_cell(poly_e, poly_n,
                     cell_e_min, cell_e_max,
                     cell_n_min, cell_n_max,
                     spacing, jitter_frac, rng):
    """Scatter grid points inside (polygon ∩ cell bbox), using the polygon for masking."""
    e_min = max(float(poly_e.min()), cell_e_min)
    e_max = min(float(poly_e.max()), cell_e_max)
    n_min = max(float(poly_n.min()), cell_n_min)
    n_max = min(float(poly_n.max()), cell_n_max)
    if e_min >= e_max or n_min >= n_max:
        return np.empty(0), np.empty(0)

    es = np.arange(e_min + spacing * 0.5, e_max, spacing)
    ns = np.arange(n_min + spacing * 0.5, n_max, spacing)
    if len(es) == 0 or len(ns) == 0:
        return np.empty(0), np.empty(0)

    ge, gn = np.meshgrid(es, ns)
    ge = ge.ravel().copy()
    gn = gn.ravel().copy()

    jitter = spacing * jitter_frac
    ge += rng.uniform(-jitter, jitter, len(ge))
    gn += rng.uniform(-jitter, jitter, len(gn))

    mask = _points_in_polygon(ge, gn, poly_e, poly_n)
    return ge[mask], gn[mask]


def _scatter_full_cell(cell_e_min, cell_e_max, cell_n_min, cell_n_max,
                       spacing, jitter_frac, rng):
    """Scatter grid points across the entire cell bbox without polygon masking."""
    es = np.arange(cell_e_min + spacing * 0.5, cell_e_max, spacing)
    ns = np.arange(cell_n_min + spacing * 0.5, cell_n_max, spacing)
    if len(es) == 0 or len(ns) == 0:
        return np.empty(0), np.empty(0)
    ge, gn = np.meshgrid(es, ns)
    ge = ge.ravel().copy()
    gn = gn.ravel().copy()
    jitter = spacing * jitter_frac
    ge += rng.uniform(-jitter, jitter, len(ge))
    gn += rng.uniform(-jitter, jitter, len(gn))
    return ge, gn


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_gltf_model(path: Path):
    import gltf
    from panda3d.core import NodePath
    node = gltf.load_model(str(path))
    return NodePath(node)


def _model_info(model_np) -> tuple[float, float]:
    """Return (natural_height_m, base_z_m).  Auto-detects centimetre exports."""
    bounds = model_np.getTightBounds()
    if not bounds:
        return 1.0, 0.0
    min_z = float(bounds[0].z)
    max_z = float(bounds[1].z)
    h = max(max_z - min_z, 1e-3)
    if h > 200.0:          # heuristic: centimetres → metres
        h    /= 100.0
        min_z /= 100.0
    return h, min_z


# ---------------------------------------------------------------------------
# Cell geometry builder
# ---------------------------------------------------------------------------

def _build_cell(trees, models, model_info, pool, offset_e, offset_n, rng):
    from panda3d.core import NodePath
    from osm3denv.fetch.models import MODEL_HEIGHTS

    parent   = NodePath("veg_cell")
    fallback = list(models.keys())

    for (e, n, z, h, vtype_key) in trees:
        candidates = list(pool.get(vtype_key, [])) + fallback
        src = None
        chosen = None
        for slug in rng.permutation(candidates):
            slug = str(slug)
            if slug in models:
                src = models[slug]
                chosen = slug
                break
        if src is None:
            continue

        nat_h, base_z = model_info.get(chosen, (1.0, 0.0))
        if chosen in MODEL_HEIGHTS:
            lo, hi_h = MODEL_HEIGHTS[chosen]
            h = float(rng.uniform(lo, hi_h))
        scale = h / nat_h if nat_h > 0.0 else 1.0

        inst = src.copyTo(parent)
        inst.setPos(e - offset_e, n - offset_n, z - base_z * scale)
        inst.setScale(scale)
        inst.setH(float(rng.integers(0, 360)))

    return parent


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Vegetation(MapEntity):
    """Camera-driven streaming vegetation from OSM polygon areas."""

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, terrain, cache_dir: Path) -> None:
        self._osm       = osm
        self._frame     = frame
        self._radius_m  = radius_m
        self._terrain   = terrain
        self._cache_dir = cache_dir

        # Populated in build()
        self._cell_polygons:      dict[tuple[int,int], list[tuple]] = {}
        self._fixed_by_cell:      dict[tuple[int,int], list[tuple]] = {}
        self._heightmap           = None
        self._grid                = 0

        # Populated in attach_to()
        self._models:      dict = {}
        self._model_info:  dict = {}
        self._veg_root     = None
        self._active_cells: dict[tuple[int,int], object] = {}
        self._first_update = True

    # ------------------------------------------------------------------
    # build() — pre-index OSM polygons per cell, no geometry yet
    # ------------------------------------------------------------------

    def build(self) -> None:
        from osm3denv.fetch.models import MODEL_POOL, fetch
        fetch(self._cache_dir)   # warn about missing files early

        td               = self._terrain.data
        self._heightmap  = td.heightmap
        self._grid       = self._heightmap.shape[0]
        r                = float(self._radius_m)
        rng              = np.random.default_rng(42)

        # --- Individual OSM tree nodes (explicit positions) ---
        tree_vtype = _VEG_TYPES["tree"]
        tree_nodes = self._osm.filter_nodes(lambda t: t.get("natural") == "tree")
        n_individual = 0
        if tree_nodes:
            lons  = np.array([nd.lon for nd in tree_nodes])
            lats  = np.array([nd.lat for nd in tree_nodes])
            e_arr, n_arr = self._frame.to_enu(lons, lats)
            z_arr = _sample_z_triangle_vec(e_arr, n_arr, self._heightmap, self._grid, r)
            valid = (np.abs(e_arr) <= r) & (np.abs(n_arr) <= r) & (z_arr >= -0.5)
            for nd, ok, e, n, z in zip(tree_nodes, valid, e_arr, n_arr, z_arr):
                if not ok:
                    continue
                try:
                    h = float(nd.tags.get("height", 0) or 0)
                    if h <= 0:
                        h = float(rng.uniform(tree_vtype.h_min, tree_vtype.h_max))
                except (ValueError, TypeError):
                    h = float(rng.uniform(tree_vtype.h_min, tree_vtype.h_max))
                ci = int(float(e) // _CELL_SIZE)
                cj = int(float(n) // _CELL_SIZE)
                self._fixed_by_cell.setdefault((ci, cj), []).append(
                    (float(e), float(n), float(z), h)
                )
                n_individual += 1

        # --- Polygon vegetation areas ---
        def _index_polygon(east: np.ndarray, north: np.ndarray, vtype_key: str) -> None:
            if len(east) < 3:
                return
            mask = (east  >= -r) & (east  <= r) & (north >= -r) & (north <= r)
            if not mask.any():
                return
            ci_min = int(east.min()  // _CELL_SIZE)
            ci_max = int(east.max()  // _CELL_SIZE)
            cj_min = int(north.min() // _CELL_SIZE)
            cj_max = int(north.max() // _CELL_SIZE)
            for ci in range(ci_min, ci_max + 1):
                for cj in range(cj_min, cj_max + 1):
                    self._cell_polygons.setdefault((ci, cj), []).append(
                        (east, north, vtype_key)
                    )

        n_poly = 0
        for way in self._osm.filter_ways(lambda t: _classify(t) is not None):
            vk   = _classify(way.tags)
            geom = way.geometry
            if len(geom) < 3:
                continue
            lons  = np.fromiter((p[0] for p in geom), np.float64, len(geom))
            lats  = np.fromiter((p[1] for p in geom), np.float64, len(geom))
            east, north = self._frame.to_enu(lons, lats)
            _index_polygon(east, north, vk)
            n_poly += 1

        for rel in self._osm.filter_relations(lambda t: _classify(t) is not None):
            vk = _classify(rel.tags)
            for role, ring in rel.rings:
                if role not in ("outer", "") or len(ring) < 3:
                    continue
                lons  = np.fromiter((p[0] for p in ring), np.float64, len(ring))
                lats  = np.fromiter((p[1] for p in ring), np.float64, len(ring))
                east, north = self._frame.to_enu(lons, lats)
                _index_polygon(east, north, vk)
                n_poly += 1

        n_cells = len(set(self._cell_polygons) | set(self._fixed_by_cell))
        log.warning("vegetation: indexed %d polygons + %d individual trees across %d cells",
                    n_poly, n_individual, n_cells)

    # ------------------------------------------------------------------
    # attach_to() — load models, register streaming task
    # ------------------------------------------------------------------

    def attach_to(self, parent) -> None:
        from osm3denv.fetch.models import MODEL_POOL, fetch

        needed_slugs = {s for pool in MODEL_POOL.values() for s in pool}
        glb_paths = fetch(self._cache_dir, list(needed_slugs))
        if not glb_paths:
            log.warning("vegetation: no models found — place GLBs in %s/evolveduk/",
                        self._cache_dir)
            return

        for slug, path in glb_paths.items():
            try:
                m = _load_gltf_model(path)
                self._models[slug] = m
                self._model_info[slug] = _model_info(m)
                h, bz = self._model_info[slug]
                log.warning("loaded %s (h=%.2fm base_z=%.2fm)", slug, h, bz)
            except Exception as exc:
                log.warning("failed to load %s: %s", path, exc)

        if not self._models:
            log.warning("vegetation: all model loads failed")
            return

        self._veg_root = parent.attachNewNode("vegetation")
        self._veg_root.setTwoSided(True)

        builtins.base.taskMgr.add(self._stream_task, "veg_stream")

    # ------------------------------------------------------------------
    # Per-frame streaming task
    # ------------------------------------------------------------------

    def _stream_task(self, task):
        pos   = builtins.base.camera.getPos()
        cam_e = float(pos.x)
        cam_n = float(pos.y)

        r_cells = int(_STREAM_RADIUS / _CELL_SIZE) + 1
        ci_cam  = int(cam_e // _CELL_SIZE)
        cj_cam  = int(cam_n // _CELL_SIZE)

        needed: set[tuple[int, int]] = set()
        for dci in range(-r_cells, r_cells + 1):
            for dcj in range(-r_cells, r_cells + 1):
                ci, cj = ci_cam + dci, cj_cam + dcj
                cx = (ci + 0.5) * _CELL_SIZE
                cn = (cj + 0.5) * _CELL_SIZE
                if math.sqrt((cx - cam_e) ** 2 + (cn - cam_n) ** 2) <= _STREAM_RADIUS:
                    needed.add((ci, cj))

        active = set(self._active_cells.keys())

        # Load new cells — unlimited on first update, capped thereafter
        to_load  = needed - active
        max_load = len(to_load) if self._first_update else _MAX_LOADS_PER_FRAME
        self._first_update = False
        for key in list(to_load)[:max_load]:
            self._load_cell(*key)

        # Unload distant cells
        for key in list(active - needed):
            ci, cj = key
            cx = (ci + 0.5) * _CELL_SIZE
            cn = (cj + 0.5) * _CELL_SIZE
            if math.sqrt((cx - cam_e) ** 2 + (cn - cam_n) ** 2) > _UNLOAD_RADIUS:
                self._unload_cell(ci, cj)

        return task.cont

    # ------------------------------------------------------------------
    # Cell load / unload
    # ------------------------------------------------------------------

    def _load_cell(self, ci: int, cj: int) -> None:
        from osm3denv.fetch.models import MODEL_POOL

        rng = np.random.default_rng(hash((ci, cj)) & 0xFFFF_FFFF)

        cell_e_min = ci * _CELL_SIZE
        cell_e_max = (ci + 1) * _CELL_SIZE
        cell_n_min = cj * _CELL_SIZE
        cell_n_max = (cj + 1) * _CELL_SIZE
        cx = (ci + 0.5) * _CELL_SIZE
        cn = (cj + 0.5) * _CELL_SIZE

        trees: list[tuple] = []

        for (poly_e, poly_n, vtype_key) in self._cell_polygons.get((ci, cj), []):
            vtype = _VEG_TYPES[vtype_key]
            se, sn = _scatter_in_cell(
                poly_e, poly_n,
                cell_e_min, cell_e_max, cell_n_min, cell_n_max,
                vtype.spacing, vtype.jitter, rng,
            )
            if len(se) == 0:
                continue
            z_vals = _sample_z_triangle_vec(
                se, sn, self._heightmap, self._grid, self._radius_m)
            h_vals = rng.uniform(vtype.h_min, vtype.h_max, len(se)).astype(np.float32)
            valid  = z_vals >= -0.5
            for i in np.where(valid)[0]:
                trees.append((float(se[i]), float(sn[i]),
                              float(z_vals[i]), float(h_vals[i]), vtype_key))

        for (e, n, z, h) in self._fixed_by_cell.get((ci, cj), []):
            trees.append((e, n, z, h, "tree"))

        # Ground-cover pass — grass_claster + daisies across the entire cell
        groundcover: list[tuple] = []
        se, sn = _scatter_full_cell(
            cell_e_min, cell_e_max, cell_n_min, cell_n_max,
            _GROUNDCOVER_SPACING, 0.5, rng,
        )
        if len(se) > 0:
            z_vals = _sample_z_triangle_vec(
                se, sn, self._heightmap, self._grid, self._radius_m)
            for i in np.where(z_vals >= -0.5)[0]:
                groundcover.append((float(se[i]), float(sn[i]),
                                    float(z_vals[i]), 0.15, "_ground_cover"))
        if len(groundcover) > _MAX_CELL_GROUNDCOVER:
            idx = rng.choice(len(groundcover), _MAX_CELL_GROUNDCOVER, replace=False)
            groundcover = [groundcover[i] for i in idx]

        if not trees and not groundcover:
            return

        if len(trees) > _MAX_CELL_TREES:
            idx = rng.choice(len(trees), _MAX_CELL_TREES, replace=False)
            trees = [trees[i] for i in idx]

        all_plants = trees + groundcover
        cell_np = _build_cell(all_plants, self._models, self._model_info,
                              MODEL_POOL, cx, cn, rng)
        cell_np.reparentTo(self._veg_root)
        cell_np.setPos(cx, cn, 0.0)
        self._active_cells[(ci, cj)] = cell_np
        log.warning("veg: loaded (%d,%d) — %d trees + %d groundcover",
                    ci, cj, len(trees), len(groundcover))

    def _unload_cell(self, ci: int, cj: int) -> None:
        np_node = self._active_cells.pop((ci, cj), None)
        if np_node is not None:
            np_node.removeNode()
            log.warning("veg: unloaded (%d,%d)", ci, cj)
