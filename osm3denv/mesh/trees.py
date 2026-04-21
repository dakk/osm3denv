"""Procedural trees at OSM natural=tree nodes.

Each tree is a trunk (hexagonal prism) plus a crown. Crown shape is chosen
from the OSM ``leaf_type`` tag, falling back to alternating species based on
the node id for visual variety.

All trees are batched into a single ManualObject. Per-vertex UV0 carries two
things for the shader: ``uv.x`` is a sway factor (0 at the base, 1 at the
crown top) and ``uv.y`` is a stable per-tree seed in [0, 1) used for wind
phase + colour variation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely.vectorized as sv

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.geom import (parse_number, polygon_from_way,
                                polygons_from_relation)
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

MAX_TREES = 5000
SIDES = 6

# Scatter density per category: jittered-grid spacing in metres (smaller =
# denser). These control how many synthetic trees get dropped into park and
# wood polygons that don't have explicit natural=tree nodes.
SCATTER_CATEGORIES: list[tuple[str, set[str], float]] = [
    ("leisure", {"park", "garden", "nature_reserve", "village_green"}, 12.0),
    ("natural", {"wood"}, 7.0),
    ("landuse", {"forest"}, 7.0),
    ("natural", {"scrub"}, 6.0),
]
# Sway factor cutoff used in trees.frag to tell trunk apart from crown.
_TRUNK_UV = 0.12
_CROWN_BASE_UV = 0.30


@dataclass
class TreesMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    count: int


def _species(tags: dict[str, str], node_id: int) -> str:
    lt = (tags.get("leaf_type") or "").lower()
    if lt in ("needleleaved", "needle", "coniferous"):
        return "conifer"
    if lt in ("broadleaved", "broadleaf", "deciduous"):
        return "broadleaf"
    cycle = (tags.get("leaf_cycle") or "").lower()
    if cycle in ("evergreen",):
        return "conifer"
    # Fallback: deterministic alternation so forests look mixed instead of
    # uniform cones.
    return "broadleaf" if (node_id & 1) == 0 else "conifer"


def _ring(east: float, north: float, y: float, radius: float,
          twist: float) -> np.ndarray:
    angles = np.linspace(0.0, 2 * np.pi, SIDES, endpoint=False) + twist
    return np.stack([east + radius * np.cos(angles),
                     np.full(SIDES, y),
                     -north + radius * np.sin(angles)], axis=-1).astype(np.float32)


def _trunk(east: float, north: float, base_y: float, trunk_h: float,
           r_bot: float, r_top: float, twist: float,
           sway_bot: float, sway_top: float, seed_uv: float):
    """Hexagonal prism trunk. Returns (verts, normals, indices, uvs)."""
    bot = _ring(east, north, base_y, r_bot, twist)                 # 6
    top = _ring(east, north, base_y + trunk_h, r_top, twist)       # 6
    verts = np.concatenate([bot, top], axis=0)                     # 12

    # Side faces, each quad as 2 tris wound outward.
    indices = []
    for i in range(SIDES):
        j = (i + 1) % SIDES
        indices.append((i, j, i + SIDES))
        indices.append((j, j + SIDES, i + SIDES))
    indices = np.array(indices, dtype=np.uint32)

    # Smooth outward normals: horizontal only, no y component.
    normals = verts.copy()
    normals[:, 0] -= east
    normals[:, 2] -= -north
    normals[:, 1] = 0.0
    lens = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-6)
    normals = (normals / lens).astype(np.float32)

    uvs = np.empty((12, 2), dtype=np.float32)
    uvs[:SIDES, 0] = sway_bot
    uvs[SIDES:, 0] = sway_top
    uvs[:, 1] = seed_uv
    return verts, normals, indices, uvs


def _conifer_crown(east: float, north: float, base_y: float,
                   crown_h: float, r: float, twist: float,
                   seed_uv: float):
    """Single cone, base at ``base_y``, apex at ``base_y + crown_h``."""
    base = _ring(east, north, base_y, r, twist)                    # 6
    apex = np.array([[east, base_y + crown_h, -north]], dtype=np.float32)
    verts = np.concatenate([base, apex], axis=0)                   # 7

    indices = []
    normals = np.zeros_like(verts)
    for i in range(SIDES):
        j = (i + 1) % SIDES
        a, b, c = base[i], base[j], apex[0]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln > 1e-6:
            n = n / ln
            normals[i] += n
            normals[j] += n
            normals[SIDES] += n
        indices.append((i, j, SIDES))
    indices = np.array(indices, dtype=np.uint32)
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-6)
    normals = (normals / norm_lens).astype(np.float32)

    uvs = np.empty((7, 2), dtype=np.float32)
    uvs[:SIDES, 0] = _CROWN_BASE_UV
    uvs[SIDES, 0] = 1.0
    uvs[:, 1] = seed_uv
    return verts, normals, indices, uvs


def _broadleaf_crown(east: float, north: float, base_y: float,
                     crown_h: float, r: float, twist: float,
                     seed_uv: float):
    """Diamond/bipyramid: bottom tip, equator ring, top tip."""
    equator_y = base_y + 0.40 * crown_h
    bottom = np.array([[east, base_y, -north]], dtype=np.float32)
    equator = _ring(east, north, equator_y, r, twist)              # 6
    top = np.array([[east, base_y + crown_h, -north]], dtype=np.float32)
    verts = np.concatenate([bottom, equator, top], axis=0)         # 8

    indices = []
    normals = np.zeros_like(verts)
    for i in range(SIDES):
        j = (i + 1) % SIDES
        # Lower tri: bottom tip, equator[j], equator[i]
        a, b, c = bottom[0], equator[j], equator[i]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln > 1e-6:
            n = n / ln
            normals[0] += n
            normals[1 + j] += n
            normals[1 + i] += n
        indices.append((0, 1 + j, 1 + i))
        # Upper tri: equator[i], equator[j], top
        a, b, c = equator[i], equator[j], top[0]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln > 1e-6:
            n = n / ln
            normals[1 + i] += n
            normals[1 + j] += n
            normals[7] += n
        indices.append((1 + i, 1 + j, 7))
    indices = np.array(indices, dtype=np.uint32)
    norm_lens = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-6)
    normals = (normals / norm_lens).astype(np.float32)

    uvs = np.empty((8, 2), dtype=np.float32)
    uvs[0, 0] = _CROWN_BASE_UV            # bottom tip touches trunk
    uvs[1:1 + SIDES, 0] = 0.55            # equator
    uvs[7, 0] = 1.0                        # top
    uvs[:, 1] = seed_uv
    return verts, normals, indices, uvs


def _tree(species: str, east: float, north: float, base_y: float,
          height: float, seed: int):
    twist = (seed % 360) * (np.pi / 180.0)
    seed_uv = ((seed * 2654435761) & 0xFFFFFF) / float(0x1000000)

    trunk_h = max(1.2, 0.28 * height)
    r_bot = 0.06 * height
    r_top = 0.04 * height
    tv, tn, ti, tu = _trunk(east, north, base_y, trunk_h,
                            r_bot, r_top, twist,
                            sway_bot=0.0, sway_top=_TRUNK_UV,
                            seed_uv=seed_uv)

    crown_base_y = base_y + trunk_h
    crown_h = height - trunk_h

    if species == "conifer":
        r = max(0.8, 0.28 * height)
        cv, cn, ci, cu = _conifer_crown(east, north, crown_base_y, crown_h,
                                        r, twist, seed_uv)
    else:  # broadleaf (default)
        r = max(1.0, 0.40 * height)
        cv, cn, ci, cu = _broadleaf_crown(east, north, crown_base_y, crown_h,
                                          r, twist, seed_uv)

    v = np.concatenate([tv, cv], axis=0)
    n = np.concatenate([tn, cn], axis=0)
    i = np.concatenate([ti.ravel(), ci.ravel() + len(tv)], axis=0).astype(np.uint32)
    u = np.concatenate([tu, cu], axis=0)
    return v, n, i, u


def _scatter_in_polygon(poly, spacing_m: float, seed_base: int,
                        radius_m: float | None):
    """Jittered grid over the polygon, keeping only interior points.

    Returns an (N, 2) float array of (east, north) scatter positions.
    """
    minx, miny, maxx, maxy = poly.bounds
    if radius_m is not None:
        minx = max(minx, -radius_m)
        maxx = min(maxx, radius_m)
        miny = max(miny, -radius_m)
        maxy = min(maxy, radius_m)
    if maxx - minx <= 0 or maxy - miny <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    nx = max(1, int(np.floor((maxx - minx) / spacing_m)))
    ny = max(1, int(np.floor((maxy - miny) / spacing_m)))
    xs = minx + (np.arange(nx) + 0.5) * spacing_m
    ys = miny + (np.arange(ny) + 0.5) * spacing_m
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    rng = np.random.default_rng(seed_base & 0xFFFFFFFF)
    pts = pts + (rng.random(pts.shape) - 0.5) * spacing_m * 0.8

    mask = sv.contains(poly, pts[:, 0], pts[:, 1])
    return pts[mask].astype(np.float32)


def _scatter_polygons(osm: OSMData, frame: Frame):
    """Yield (polygon, spacing_m, seed) for every scatter-eligible OSM area."""
    for way in osm.ways:
        for key, values, spacing in SCATTER_CATEGORIES:
            if way.tags.get(key) in values:
                poly = polygon_from_way(way, frame, min_area=50.0)
                if poly is not None:
                    yield poly, spacing, way.id
                break
    for rel in osm.relations:
        for key, values, spacing in SCATTER_CATEGORIES:
            if rel.tags.get(key) in values:
                for poly in polygons_from_relation(rel, frame, min_area=50.0):
                    yield poly, spacing, rel.id
                break


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler,
          *, radius_m: float | None = None) -> TreesMesh:
    empty = TreesMesh(
        vertices=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        indices=np.zeros((0,), dtype=np.uint32),
        uvs=np.zeros((0, 2), dtype=np.float32),
        count=0,
    )

    all_v: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    all_u: list[np.ndarray] = []
    v_off = 0
    count = 0

    def _append(v, nrm, idx, uv):
        nonlocal v_off, count
        all_v.append(v)
        all_n.append(nrm)
        all_i.append(idx + v_off)
        all_u.append(uv)
        v_off += len(v)
        count += 1

    # --- Pass 1: real trees from natural=tree nodes (prioritised). --------
    tree_nodes = osm.filter_nodes(lambda t: t.get("natural") == "tree")
    if tree_nodes:
        lon = np.asarray([n.lon for n in tree_nodes], dtype=np.float64)
        lat = np.asarray([n.lat for n in tree_nodes], dtype=np.float64)
        east, north = frame.to_enu(lon, lat)

        if radius_m is not None:
            inside = (np.abs(east) <= radius_m) & (np.abs(north) <= radius_m)
        else:
            inside = np.ones_like(east, dtype=bool)

        idxs = np.flatnonzero(inside)
        if len(idxs) > MAX_TREES:
            dist = np.hypot(east[idxs], north[idxs])
            keep = np.argsort(dist)[:MAX_TREES]
            idxs = idxs[keep]
            log.warning("trees: %d within radius, keeping nearest %d",
                        int(inside.sum()), MAX_TREES)

        for i in idxs:
            node = tree_nodes[int(i)]
            e, n = float(east[int(i)]), float(north[int(i)])
            base_y = float(sampler.height_at(e, n))
            h = parse_number(node.tags.get("height"))
            if h is None or h <= 0:
                h = 6.0 + (node.id % 5)
            species = _species(node.tags, node.id)
            v, nrm, idx, uv = _tree(species, e, n, base_y, float(h), node.id)
            _append(v, nrm, idx, uv)

    # --- Pass 2: synthetic trees scattered inside park / wood polygons ----
    remaining = MAX_TREES - count
    if remaining > 0:
        scatter_count = 0
        for poly, spacing, seed in _scatter_polygons(osm, frame):
            if remaining <= 0:
                break
            pts = _scatter_in_polygon(poly, spacing, seed, radius_m)
            if len(pts) == 0:
                continue
            if len(pts) > remaining:
                # Keep a deterministic subset (first N after jitter).
                pts = pts[:remaining]
            rng = np.random.default_rng((seed * 1315423911) & 0xFFFFFFFF)
            heights = rng.uniform(5.0, 11.0, size=len(pts))
            for (e, n), h in zip(pts, heights):
                e = float(e)
                n = float(n)
                base_y = float(sampler.height_at(e, n))
                # Derive a stable per-tree id from (polygon seed, scatter index)
                node_id = (seed * 1000003 + scatter_count) & 0x7FFFFFFF
                species = _species({}, node_id)
                v, nrm, idx, uv = _tree(species, e, n, base_y, float(h), node_id)
                _append(v, nrm, idx, uv)
                scatter_count += 1
                remaining -= 1
                if remaining <= 0:
                    break
        if scatter_count:
            log.info("trees: scattered %d synthetic trees inside park/wood polygons",
                     scatter_count)

    if not all_v:
        return empty
    return TreesMesh(
        vertices=np.concatenate(all_v, axis=0).astype(np.float32),
        normals=np.concatenate(all_n, axis=0).astype(np.float32),
        indices=np.concatenate(all_i, axis=0).astype(np.uint32),
        uvs=np.concatenate(all_u, axis=0).astype(np.float32),
        count=count,
    )
