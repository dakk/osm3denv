"""Vegetation entity — individual trees and scattered polygon vegetation.

Each OSM tag type (forest, park, scrub, orchard, etc.) is described by a
VegType dataclass with its own height range, crown proportions, scatter
density, and colour.  Trees are rendered as two crossed vertical quads
(cross-billboard) batched into a single GeomNode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LOD parameters
# ---------------------------------------------------------------------------
_LOD_CELL_SIZE = 200.0   # spatial grid cell side length (m)
_LOD_NEAR      = 600.0   # full-detail switch distance (m from cell centre)
_LOD_FAR       = 2000.0  # reduced-detail switch distance (m); invisible beyond
_LOD_LOW_STEP  = 3       # keep every Nth tree in the reduced-detail level


@dataclass(frozen=True)
class VegType:
    h_min:        float                                # min tree height (m)
    h_max:        float                                # max tree height (m)
    width_ratio:  float                                # crown width = height × ratio
    spacing:      float                                # scatter grid spacing (m)
    jitter:       float                                # jitter as fraction of spacing
    budget:       int                                  # max trees for this type
    color_top:    tuple[float, float, float, float]    # RGBA crown colour
    color_bottom: tuple[float, float, float, float]    # RGBA trunk base colour


_VEG_TYPES: dict[str, VegType] = {
    # Open parkland — deciduous, wide crowns, well-spaced
    "park": VegType(
        h_min=7.0, h_max=14.0, width_ratio=0.85, spacing=28.0, jitter=0.50,
        budget=15_000,
        color_top   =(0.22, 0.55, 0.15, 1.0),
        color_bottom=(0.12, 0.28, 0.08, 1.0),
    ),
    # Fruit trees — medium height, rounded crown, near-regular grid
    "orchard": VegType(
        h_min=3.0, h_max=6.0, width_ratio=0.95, spacing=8.0, jitter=0.08,
        budget=8_000,
        color_top   =(0.26, 0.52, 0.14, 1.0),
        color_bottom=(0.14, 0.26, 0.07, 1.0),
    ),
    # Dense low shrubs — wider than tall, olive-green
    "scrub": VegType(
        h_min=1.5, h_max=4.0, width_ratio=1.30, spacing=6.0, jitter=0.50,
        budget=30_000,
        color_top   =(0.30, 0.42, 0.15, 1.0),
        color_bottom=(0.18, 0.25, 0.10, 1.0),
    ),
    # Very short heather / gorse — brownish, sparse
    "heath": VegType(
        h_min=0.6, h_max=1.5, width_ratio=1.60, spacing=5.0, jitter=0.60,
        budget=15_000,
        color_top   =(0.38, 0.32, 0.18, 1.0),
        color_bottom=(0.22, 0.18, 0.10, 1.0),
    ),
    # Tall narrow conifers (yew / cypress) — columnar, dark green
    "cemetery": VegType(
        h_min=8.0, h_max=16.0, width_ratio=0.40, spacing=18.0, jitter=0.30,
        budget=2_000,
        color_top   =(0.08, 0.28, 0.12, 1.0),
        color_bottom=(0.04, 0.14, 0.06, 1.0),
    ),
    # Small ornamental / suburban garden trees
    "garden": VegType(
        h_min=3.0, h_max=7.0, width_ratio=0.80, spacing=12.0, jitter=0.50,
        budget=6_000,
        color_top   =(0.24, 0.52, 0.14, 1.0),
        color_bottom=(0.12, 0.28, 0.08, 1.0),
    ),
    # Specimen trees on village greens — large, spreading crown
    "village_green": VegType(
        h_min=8.0, h_max=14.0, width_ratio=1.00, spacing=25.0, jitter=0.40,
        budget=1_000,
        color_top   =(0.25, 0.55, 0.16, 1.0),
        color_bottom=(0.13, 0.28, 0.08, 1.0),
    ),
    # Small fruit / vegetable-plot trees in allotment gardens
    "allotments": VegType(
        h_min=2.0, h_max=4.0, width_ratio=0.85, spacing=8.0, jitter=0.30,
        budget=4_000,
        color_top   =(0.28, 0.50, 0.16, 1.0),
        color_bottom=(0.14, 0.26, 0.08, 1.0),
    ),
    # Default for individual natural=tree OSM nodes
    "tree": VegType(
        h_min=6.0, h_max=12.0, width_ratio=0.80, spacing=0.0, jitter=0.0,
        budget=5_000,
        color_top   =(0.22, 0.52, 0.14, 1.0),
        color_bottom=(0.12, 0.26, 0.07, 1.0),
    ),
    # Dense closed-canopy woodland — processed last so other types are not crowded out
    "forest": VegType(
        h_min=10.0, h_max=18.0, width_ratio=0.65, spacing=20.0, jitter=0.40,
        budget=50_000,
        color_top   =(0.13, 0.38, 0.08, 1.0),
        color_bottom=(0.07, 0.18, 0.04, 1.0),
    ),
}


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
    return None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sample_z_triangle_vec(e_arr: np.ndarray, n_arr: np.ndarray,
                           heightmap: np.ndarray, grid: int,
                           radius_m: float) -> np.ndarray:
    """Vectorized triangle-aware terrain z for arrays of (east, north) points.

    Matches the GPU SW→NE diagonal tessellation exactly (see scalar version
    for derivation).
    """
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


def _points_in_polygon(pts_e: np.ndarray, pts_n: np.ndarray,
                       poly_e: np.ndarray, poly_n: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon test."""
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


def _scatter_in_polygon(poly_e: np.ndarray, poly_n: np.ndarray,
                        spacing: float, jitter_frac: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (east, north) scatter points inside a polygon."""
    e_min, e_max = poly_e.min(), poly_e.max()
    n_min, n_max = poly_n.min(), poly_n.max()
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


# ---------------------------------------------------------------------------
# Geometry builder — vectorised via numpy bulk copy
# ---------------------------------------------------------------------------

def _build_cell_geom(trees: list, offset_e: float, offset_n: float):
    """Vectorized cross-quad GeomNode built via numpy bulk memory copy."""
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles,
        GeomVertexData, GeomVertexFormat,
    )

    n = len(trees)
    if n == 0:
        return GeomNode("veg_cell")

    fmt   = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("veg", fmt, Geom.UHStatic)
    vdata.unclean_set_num_rows(n * 8)

    # --- Per-tree scalar arrays ---
    e_a = np.fromiter((t[0] for t in trees), np.float32, n) - np.float32(offset_e)
    n_a = np.fromiter((t[1] for t in trees), np.float32, n) - np.float32(offset_n)
    z_a = np.fromiter((t[2] for t in trees), np.float32, n)
    h_a = np.fromiter((t[3] for t in trees), np.float32, n)
    top = z_a + h_a
    hw  = h_a * np.fromiter((t[4].width_ratio for t in trees), np.float32, n) * 0.5

    # Float 0-1 colours → uint8 0-255, shape (n, 4)
    ct = (np.array([t[4].color_top    for t in trees], np.float32) * 255).astype(np.uint8)
    cb = (np.array([t[4].color_bottom for t in trees], np.float32) * 255).astype(np.uint8)

    # --- Vertex buffer ---
    # getV3c4() layout: 3×float32 (12 B) + 4×uint8 (4 B) = 16 B/vertex, no padding
    dt   = np.dtype([('xyz', np.float32, 3), ('rgba', np.uint8, 4)])
    vbuf = np.empty(n * 8, dtype=dt)

    # 8 vertex slots per tree using strided assignment
    vbuf['xyz'][0::8] = np.stack([e_a - hw, n_a,       z_a], 1)
    vbuf['xyz'][1::8] = np.stack([e_a + hw, n_a,       z_a], 1)
    vbuf['xyz'][2::8] = np.stack([e_a + hw, n_a,       top], 1)
    vbuf['xyz'][3::8] = np.stack([e_a - hw, n_a,       top], 1)
    vbuf['xyz'][4::8] = np.stack([e_a,      n_a - hw,  z_a], 1)
    vbuf['xyz'][5::8] = np.stack([e_a,      n_a + hw,  z_a], 1)
    vbuf['xyz'][6::8] = np.stack([e_a,      n_a + hw,  top], 1)
    vbuf['xyz'][7::8] = np.stack([e_a,      n_a - hw,  top], 1)

    for slot in (0, 1, 4, 5):   # trunk/base colour
        vbuf['rgba'][slot::8] = cb
    for slot in (2, 3, 6, 7):   # crown colour
        vbuf['rgba'][slot::8] = ct

    vdata.modify_array(0).modify_handle().copy_data_from(vbuf.tobytes())

    # --- Index buffer ---
    # 4 triangles per tree (Q0: 0,1,2 + 0,2,3 ; Q1: 4,5,6 + 4,6,7) = 12 indices
    # Max vertex index = n*8-1; n ≤ a few hundred per cell → uint16 is safe
    offsets = np.array([0, 1, 2,  0, 2, 3,  4, 5, 6,  4, 6, 7], dtype=np.uint16)
    bases   = (np.arange(n, dtype=np.uint16) * np.uint16(8))[:, np.newaxis]
    idx     = (bases + offsets).reshape(-1)

    tris = GeomTriangles(Geom.UHStatic)
    idx_arr = tris.modify_vertices()
    idx_arr.unclean_set_num_rows(int(idx.shape[0]))
    idx_arr.modify_handle().copy_data_from(idx.tobytes())
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode("veg_cell")
    node.addGeom(geom)
    return node


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Vegetation(MapEntity):
    """Cross-billboard trees placed from OSM nodes and polygon areas."""

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, terrain) -> None:
        self._osm      = osm
        self._frame    = frame
        self._radius_m = radius_m
        self._terrain  = terrain
        # (east, north, z, height, VegType)
        self._trees: list[tuple] = []

    def build(self) -> None:
        td        = self._terrain.data
        heightmap = td.heightmap
        grid      = heightmap.shape[0]
        r         = float(self._radius_m)
        rng       = np.random.default_rng(42)

        # --- 1. Individual OSM tree nodes (batch to_enu + z-sampling) ---
        tree_vtype = _VEG_TYPES["tree"]
        tree_nodes = self._osm.filter_nodes(lambda t: t.get("natural") == "tree")
        n_individual = 0
        if tree_nodes:
            lons = np.array([nd.lon for nd in tree_nodes])
            lats = np.array([nd.lat for nd in tree_nodes])
            e_arr, n_arr = self._frame.to_enu(lons, lats)
            z_arr = _sample_z_triangle_vec(e_arr, n_arr, heightmap, grid, r)
            valid = (np.abs(e_arr) <= r) & (np.abs(n_arr) <= r) & (z_arr >= -0.5)
            h_list = []
            for nd in tree_nodes:
                try:
                    h = float(nd.tags.get("height", 0) or 0)
                    if h <= 0:
                        h = float(rng.uniform(tree_vtype.h_min, tree_vtype.h_max))
                except (ValueError, TypeError):
                    h = float(rng.uniform(tree_vtype.h_min, tree_vtype.h_max))
                h_list.append(h)
            h_arr = np.array(h_list, dtype=np.float32)
            for i in np.where(valid)[0][:tree_vtype.budget]:
                self._trees.append((float(e_arr[i]), float(n_arr[i]),
                                    float(z_arr[i]), float(h_arr[i]), tree_vtype))
                n_individual += 1

        # --- 2 & 3. Vegetation polygon areas ---
        type_used: dict[str, int] = {}

        def scatter_polygon(poly_e: np.ndarray, poly_n: np.ndarray,
                            vtype_key: str, vtype: VegType) -> None:
            if len(poly_e) < 3:
                return
            budget_left = vtype.budget - type_used.get(vtype_key, 0)
            if budget_left <= 0:
                return
            se, sn = _scatter_in_polygon(
                poly_e, poly_n, vtype.spacing, vtype.jitter, rng)
            if len(se) == 0:
                return
            z_vals = _sample_z_triangle_vec(se, sn, heightmap, grid, r)
            h_vals = rng.uniform(vtype.h_min, vtype.h_max, len(se)).astype(np.float32)
            mask   = (np.abs(se) <= r) & (np.abs(sn) <= r) & (z_vals >= -0.5)
            se, sn, z_vals, h_vals = (
                se[mask], sn[mask], z_vals[mask], h_vals[mask])
            n_add = min(len(se), budget_left)
            self._trees.extend(
                (float(se[i]), float(sn[i]), float(z_vals[i]),
                 float(h_vals[i]), vtype) for i in range(n_add)
            )
            type_used[vtype_key] = type_used.get(vtype_key, 0) + n_add

        for way in self._osm.filter_ways(lambda t: _classify(t) is not None):
            vtype_key = _classify(way.tags)
            vtype     = _VEG_TYPES[vtype_key]          # type: ignore[index]
            geom      = way.geometry
            if len(geom) < 3:
                continue
            lons  = np.fromiter((p[0] for p in geom), np.float64, len(geom))
            lats  = np.fromiter((p[1] for p in geom), np.float64, len(geom))
            east, north = self._frame.to_enu(lons, lats)
            if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                continue
            scatter_polygon(east, north, vtype_key, vtype)

        for rel in self._osm.filter_relations(lambda t: _classify(t) is not None):
            vtype_key = _classify(rel.tags)
            vtype     = _VEG_TYPES[vtype_key]          # type: ignore[index]
            for role, ring in rel.rings:
                if role not in ("outer", ""):
                    continue
                if len(ring) < 3:
                    continue
                lons  = np.fromiter((p[0] for p in ring), np.float64, len(ring))
                lats  = np.fromiter((p[1] for p in ring), np.float64, len(ring))
                east, north = self._frame.to_enu(lons, lats)
                if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                    continue
                scatter_polygon(east, north, vtype_key, vtype)

        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(type_used.items()))
        log.info("vegetation: %d individual + areas (%s) → %d trees total",
                 n_individual, counts_str, len(self._trees))

    def attach_to(self, parent) -> None:
        if not self._trees:
            return

        from panda3d.core import LODNode
        from osm3denv.render.helpers import load_shader

        # One root node — shader and render state cascade to all LOD cells.
        veg_root = parent.attachNewNode("vegetation")
        veg_root.setTwoSided(True)
        veg_root.setLightOff()
        shader = load_shader("vegetation")
        if shader:
            veg_root.setShader(shader)

        # Group trees into spatial grid cells.
        cells: dict[tuple[int, int], list] = {}
        for tree in self._trees:
            e, n = tree[0], tree[1]
            key = (int(e // _LOD_CELL_SIZE), int(n // _LOD_CELL_SIZE))
            cells.setdefault(key, []).append(tree)

        # One LODNode per cell with two detail levels.
        for (ci, cj), trees in cells.items():
            cx = (ci + 0.5) * _LOD_CELL_SIZE
            cn = (cj + 0.5) * _LOD_CELL_SIZE

            lod_np = veg_root.attachNewNode(LODNode(f"veg_{ci}_{cj}"))
            lod_np.setPos(cx, cn, 0.0)

            # High detail: all trees — visible within _LOD_NEAR.
            lod_np.attachNewNode(_build_cell_geom(trees, cx, cn))
            lod_np.node().addSwitch(_LOD_NEAR, 0.0)

            # Low detail: every Nth tree — visible _LOD_NEAR … _LOD_FAR.
            lod_np.attachNewNode(_build_cell_geom(trees[::_LOD_LOW_STEP], cx, cn))
            lod_np.node().addSwitch(_LOD_FAR, _LOD_NEAR)
            # Beyond _LOD_FAR no child matches → cell is invisible.

        log.info("vegetation: %d trees in %d LOD cells", len(self._trees), len(cells))
