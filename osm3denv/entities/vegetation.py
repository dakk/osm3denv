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
from osm3denv.entities.utils import grid_coords
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_MAX_TREES_PER_TYPE = 20_000

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
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.22, 0.55, 0.15, 1.0),
        color_bottom=(0.12, 0.28, 0.08, 1.0),
    ),
    # Fruit trees — medium height, rounded crown, near-regular grid
    "orchard": VegType(
        h_min=3.0, h_max=6.0, width_ratio=0.95, spacing=8.0, jitter=0.08,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.26, 0.52, 0.14, 1.0),
        color_bottom=(0.14, 0.26, 0.07, 1.0),
    ),
    # Dense low shrubs — wider than tall, olive-green
    "scrub": VegType(
        h_min=1.5, h_max=4.0, width_ratio=1.30, spacing=6.0, jitter=0.50,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.30, 0.42, 0.15, 1.0),
        color_bottom=(0.18, 0.25, 0.10, 1.0),
    ),
    # Very short heather / gorse — brownish, sparse
    "heath": VegType(
        h_min=0.6, h_max=1.5, width_ratio=1.60, spacing=5.0, jitter=0.60,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.38, 0.32, 0.18, 1.0),
        color_bottom=(0.22, 0.18, 0.10, 1.0),
    ),
    # Tall narrow conifers (yew / cypress) — columnar, dark green
    "cemetery": VegType(
        h_min=8.0, h_max=16.0, width_ratio=0.40, spacing=18.0, jitter=0.30,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.08, 0.28, 0.12, 1.0),
        color_bottom=(0.04, 0.14, 0.06, 1.0),
    ),
    # Small ornamental / suburban garden trees
    "garden": VegType(
        h_min=3.0, h_max=7.0, width_ratio=0.80, spacing=12.0, jitter=0.50,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.24, 0.52, 0.14, 1.0),
        color_bottom=(0.12, 0.28, 0.08, 1.0),
    ),
    # Specimen trees on village greens — large, spreading crown
    "village_green": VegType(
        h_min=8.0, h_max=14.0, width_ratio=1.00, spacing=25.0, jitter=0.40,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.25, 0.55, 0.16, 1.0),
        color_bottom=(0.13, 0.28, 0.08, 1.0),
    ),
    # Small fruit / vegetable-plot trees in allotment gardens
    "allotments": VegType(
        h_min=2.0, h_max=4.0, width_ratio=0.85, spacing=8.0, jitter=0.30,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.28, 0.50, 0.16, 1.0),
        color_bottom=(0.14, 0.26, 0.08, 1.0),
    ),
    # Default for individual natural=tree OSM nodes
    "tree": VegType(
        h_min=6.0, h_max=12.0, width_ratio=0.80, spacing=0.0, jitter=0.0,
        budget=_MAX_TREES_PER_TYPE,
        color_top   =(0.22, 0.52, 0.14, 1.0),
        color_bottom=(0.12, 0.26, 0.07, 1.0),
    ),
    # Dense closed-canopy woodland — processed last so other types are not crowded out
    "forest": VegType(
        h_min=10.0, h_max=18.0, width_ratio=0.65, spacing=20.0, jitter=0.40,
        budget=_MAX_TREES_PER_TYPE,
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

def _sample_z_triangle(x: float, y: float,
                       heightmap: np.ndarray, grid: int, radius_m: float) -> float:
    """Return terrain surface z using the same triangle tessellation as the GPU.

    The terrain mesh splits each heightmap cell into two triangles along the
    SW→NE diagonal.  Bilinear interpolation diverges from linear-in-triangle
    on saddle-shaped cells, placing trees below the visual surface.  This
    function matches the GPU exactly.
    """
    row_f, col_f = grid_coords(x, y, grid, radius_m)
    r0 = min(int(row_f), grid - 2)
    c0 = min(int(col_f), grid - 2)
    fr = row_f - r0   # south fraction within cell (0 = north edge)
    fc = col_f - c0   # east  fraction within cell (0 = west edge)

    z_nw = float(heightmap[r0,     c0    ])
    z_ne = float(heightmap[r0,     c0 + 1])
    z_sw = float(heightmap[r0 + 1, c0    ])
    z_se = float(heightmap[r0 + 1, c0 + 1])

    # Diagonal from SW(fr=1,fc=0) to NE(fr=0,fc=1): fr + fc == 1
    if fr + fc >= 1.0:   # lower-right triangle: SW, SE, NE
        return (fc + fr - 1.0) * z_se + (1.0 - fr) * z_ne + (1.0 - fc) * z_sw
    else:                # upper-left  triangle: SW, NE, NW
        return fc * z_ne + (1.0 - fr - fc) * z_nw + fr * z_sw


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
        self._trees: list[tuple[float, float, float, float, VegType]] = []

    def build(self) -> None:
        td        = self._terrain.data
        heightmap = td.heightmap
        grid      = heightmap.shape[0]
        r         = float(self._radius_m)
        rng       = np.random.default_rng(42)

        def z_at(e: float, n: float) -> float:
            return _sample_z_triangle(e, n, heightmap, grid, r)

        def add_tree(e: float, n: float, h: float, vtype: VegType) -> None:
            if abs(e) > r or abs(n) > r:
                return
            z = z_at(e, n)
            if z < -0.5:
                return
            self._trees.append((e, n, z, h, vtype))

        # --- 1. Individual OSM tree nodes ---
        tree_vtype  = _VEG_TYPES["tree"]
        n_individual = 0
        for node in self._osm.filter_nodes(lambda t: t.get("natural") == "tree"):
            e_arr, n_arr = self._frame.to_enu(
                np.array([node.lon]), np.array([node.lat]))
            try:
                h = float(node.tags.get("height", 0) or 0)
                if h <= 0:
                    h = float(rng.uniform(tree_vtype.h_min, tree_vtype.h_max))
            except (ValueError, TypeError):
                h = float(rng.uniform(tree_vtype.h_min, tree_vtype.h_max))
            add_tree(float(e_arr[0]), float(n_arr[0]), h, tree_vtype)
            n_individual += 1

        type_used: dict[str, int] = {}

        def scatter_polygon(poly_e: np.ndarray, poly_n: np.ndarray,
                            vtype_key: str, vtype: VegType) -> None:
            if len(poly_e) < 3:
                return
            if type_used.get(vtype_key, 0) >= vtype.budget:
                return
            se, sn = _scatter_in_polygon(
                poly_e, poly_n, vtype.spacing, vtype.jitter, rng)
            h_vals = rng.uniform(vtype.h_min, vtype.h_max, len(se))
            for e, n, h in zip(se, sn, h_vals):
                if type_used.get(vtype_key, 0) >= vtype.budget:
                    break
                add_tree(float(e), float(n), float(h), vtype)
                type_used[vtype_key] = type_used.get(vtype_key, 0) + 1

        # --- 2. Vegetation polygon ways ---
        for way in self._osm.filter_ways(lambda t: _classify(t) is not None):
            vtype_key = _classify(way.tags)
            vtype     = _VEG_TYPES[vtype_key]          # type: ignore[index]
            geom      = way.geometry
            if len(geom) < 3:
                continue
            lons  = np.fromiter((p[0] for p in geom), np.float64, count=len(geom))
            lats  = np.fromiter((p[1] for p in geom), np.float64, count=len(geom))
            east, north = self._frame.to_enu(lons, lats)
            if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                continue
            scatter_polygon(east, north, vtype_key, vtype)

        # --- 3. Vegetation polygon relations ---
        for rel in self._osm.filter_relations(lambda t: _classify(t) is not None):
            vtype_key = _classify(rel.tags)
            vtype     = _VEG_TYPES[vtype_key]          # type: ignore[index]
            for role, ring in rel.rings:
                if role not in ("outer", ""):
                    continue
                if len(ring) < 3:
                    continue
                lons  = np.fromiter((p[0] for p in ring), np.float64, count=len(ring))
                lats  = np.fromiter((p[1] for p in ring), np.float64, count=len(ring))
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

        from panda3d.core import (
            Geom, GeomNode, GeomTriangles,
            GeomVertexData, GeomVertexFormat, GeomVertexWriter,
        )

        n_trees = len(self._trees)
        fmt   = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("vegetation", fmt, Geom.UHStatic)
        vdata.setNumRows(n_trees * 8)

        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHStatic)

        for idx, (e, n, z, h, vtype) in enumerate(self._trees):
            ct  = vtype.color_top
            cb  = vtype.color_bottom
            hw  = h * vtype.width_ratio * 0.5
            top = z + h
            b   = idx * 8

            # Quad 1: N-S plane (extends along east axis)
            vw.addData3(e - hw, n, z);   cw.addData4(*cb)
            vw.addData3(e + hw, n, z);   cw.addData4(*cb)
            vw.addData3(e + hw, n, top); cw.addData4(*ct)
            vw.addData3(e - hw, n, top); cw.addData4(*ct)

            # Quad 2: E-W plane (extends along north axis)
            vw.addData3(e, n - hw, z);   cw.addData4(*cb)
            vw.addData3(e, n + hw, z);   cw.addData4(*cb)
            vw.addData3(e, n + hw, top); cw.addData4(*ct)
            vw.addData3(e, n - hw, top); cw.addData4(*ct)

            for q in range(2):
                v = b + q * 4
                tris.addVertices(v,     v + 1, v + 2)
                tris.addVertices(v,     v + 2, v + 3)

        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("vegetation")
        node.addGeom(geom)

        np_ = parent.attachNewNode(node)
        np_.setTwoSided(True)
        np_.setLightOff()

        from osm3denv.render.helpers import load_shader
        shader = load_shader("vegetation")
        if shader:
            np_.setShader(shader)
