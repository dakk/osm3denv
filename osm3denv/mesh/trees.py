"""Low-poly tree cones at OSM natural=tree nodes.

All trees are batched into a single ManualObject. Caps at ``MAX_TREES`` to keep
upload time bounded on dense urban scenes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.geom import parse_number
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

MAX_TREES = 5000
SIDES = 6


@dataclass
class TreesMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    count: int


def _cone(east: float, north: float, base_y: float,
          height: float, radius: float, seed: int):
    # Tiny per-tree twist for variety without changing mesh count.
    twist = (seed % 360) * (np.pi / 180.0)
    angles = np.linspace(0.0, 2 * np.pi, SIDES, endpoint=False) + twist
    ring = np.stack([radius * np.cos(angles),
                     np.zeros(SIDES),
                     radius * np.sin(angles)], axis=-1)  # (SIDES, 3) local
    origin = np.array([east, base_y, -north], dtype=np.float32)
    base_ring = ring.astype(np.float32) + origin
    apex = origin + np.array([0.0, height, 0.0], dtype=np.float32)

    verts = np.concatenate([base_ring, apex[None, :]], axis=0)  # (SIDES+1, 3)

    # Side faces (each triangle has its own face normal so shading is flat).
    # Winding: (base[i], base[i+1], apex) is CCW when viewed from outside (+radial).
    indices = []
    normals = np.zeros_like(verts)
    # We'll compute per-triangle face normals and accumulate into vertex normals
    # for a smooth shaded cone.
    for i in range(SIDES):
        i2 = (i + 1) % SIDES
        a, b, c = base_ring[i], base_ring[i2], apex
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln > 1e-6:
            n = n / ln
            normals[i] += n
            normals[i2] += n
            normals[SIDES] += n
        indices.append((i, i2, SIDES))
    # Base: a fan closing the bottom. Normal = -Y (downward).
    # Not strictly needed if we never look from below, but keep the mesh watertight.
    for i in range(1, SIDES - 1):
        indices.append((0, i + 1, i))  # reversed winding so normal points -Y
    # Base vertex normals get a small -Y component; renormalize.
    for i in range(SIDES):
        normals[i] += np.array([0.0, -0.25, 0.0])
    norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-6)
    normals = normals / norm_lengths

    return verts, normals.astype(np.float32), np.array(indices, dtype=np.uint32)


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler,
          *, radius_m: float | None = None) -> TreesMesh:
    tree_nodes = osm.filter_nodes(lambda t: t.get("natural") == "tree")
    if not tree_nodes:
        return TreesMesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
            count=0,
        )

    # Project all nodes, filter to scene radius, sort by distance if we'll cap.
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

    all_v: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    v_off = 0
    count = 0
    for i in idxs:
        node = tree_nodes[int(i)]
        e, n = float(east[int(i)]), float(north[int(i)])
        base_y = float(sampler.height_at(e, n))
        h = parse_number(node.tags.get("height"))
        if h is None or h <= 0:
            h = 6.0 + (node.id % 5)
        radius = max(0.6, h * 0.20)
        v, nrm, idx = _cone(e, n, base_y, float(h), float(radius), node.id)
        all_v.append(v)
        all_n.append(nrm)
        all_i.append(idx.ravel() + v_off)
        v_off += len(v)
        count += 1

    if not all_v:
        return TreesMesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0,), dtype=np.uint32),
            count=0,
        )
    return TreesMesh(
        vertices=np.concatenate(all_v, axis=0).astype(np.float32),
        normals=np.concatenate(all_n, axis=0).astype(np.float32),
        indices=np.concatenate(all_i, axis=0).astype(np.uint32),
        count=count,
    )
