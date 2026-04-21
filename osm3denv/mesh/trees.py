"""Tree placement records: OSM ``natural=tree`` nodes + scattered synthetic
trees inside park / wood / scrub polygons.

The output is no longer geometry — just a list of placements (world position,
height, rotation, species hint). The renderer instantiates a real mesh Entity
per placement at attach time, picking from the Shapespark Low-Poly Plants Kit.
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

SCATTER_CATEGORIES: list[tuple[str, set[str], float]] = [
    ("leisure", {"park", "garden", "nature_reserve", "village_green"}, 12.0),
    ("natural", {"wood"}, 7.0),
    ("landuse", {"forest"}, 7.0),
    ("natural", {"scrub"}, 6.0),
]


@dataclass
class TreePlacement:
    east: float
    north: float
    base_y: float
    height: float            # desired world-space height in metres
    yaw_rad: float           # rotation around Y
    species: str             # "conifer" | "broadleaf" | "generic"
    seed: int                # stable per-tree id for deterministic choices


@dataclass
class TreeRecords:
    placements: list[TreePlacement]

    @property
    def count(self) -> int:
        return len(self.placements)


def _species(tags: dict[str, str], node_id: int) -> str:
    lt = (tags.get("leaf_type") or "").lower()
    if lt in ("needleleaved", "needle", "coniferous"):
        return "conifer"
    if lt in ("broadleaved", "broadleaf", "deciduous"):
        return "broadleaf"
    cycle = (tags.get("leaf_cycle") or "").lower()
    if cycle in ("evergreen",):
        return "conifer"
    return "broadleaf" if (node_id & 1) == 0 else "conifer"


def _scatter_in_polygon(poly, spacing_m: float, seed_base: int,
                        radius_m: float | None):
    minx, miny, maxx, maxy = poly.bounds
    if radius_m is not None:
        minx = max(minx, -radius_m); maxx = min(maxx, radius_m)
        miny = max(miny, -radius_m); maxy = min(maxy, radius_m)
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
          *, radius_m: float | None = None) -> TreeRecords:
    placements: list[TreePlacement] = []

    # --- Pass 1: real trees from natural=tree nodes. --------------------
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
            idxs = idxs[np.argsort(dist)[:MAX_TREES]]
            log.warning("trees: %d within radius, keeping nearest %d",
                        int(inside.sum()), MAX_TREES)

        for i in idxs:
            node = tree_nodes[int(i)]
            e = float(east[int(i)]); n = float(north[int(i)])
            base_y = float(sampler.height_at(e, n))
            h = parse_number(node.tags.get("height"))
            if h is None or h <= 0:
                h = 6.0 + (node.id % 5)
            yaw = ((node.id * 2654435761) & 0xFFFF) / 0xFFFF * 2.0 * np.pi
            placements.append(TreePlacement(
                east=e, north=n, base_y=base_y,
                height=float(h), yaw_rad=float(yaw),
                species=_species(node.tags, node.id),
                seed=int(node.id),
            ))

    # --- Pass 2: synthetic trees scattered in park/wood polygons. -------
    remaining = MAX_TREES - len(placements)
    scatter_count = 0
    if remaining > 0:
        for poly, spacing, seed in _scatter_polygons(osm, frame):
            if remaining <= 0:
                break
            pts = _scatter_in_polygon(poly, spacing, seed, radius_m)
            if len(pts) == 0:
                continue
            if len(pts) > remaining:
                pts = pts[:remaining]
            rng = np.random.default_rng((seed * 1315423911) & 0xFFFFFFFF)
            heights = rng.uniform(5.0, 11.0, size=len(pts))
            yaws = rng.uniform(0.0, 2.0 * np.pi, size=len(pts))
            for (e, n), h, yaw in zip(pts, heights, yaws):
                e = float(e); n = float(n)
                node_id = (seed * 1000003 + scatter_count) & 0x7FFFFFFF
                placements.append(TreePlacement(
                    east=e, north=n,
                    base_y=float(sampler.height_at(e, n)),
                    height=float(h), yaw_rad=float(yaw),
                    species=_species({}, node_id),
                    seed=node_id,
                ))
                scatter_count += 1
                remaining -= 1
                if remaining <= 0:
                    break
        if scatter_count:
            log.info("trees: scattered %d synthetic trees inside park/wood polygons",
                     scatter_count)

    return TreeRecords(placements=placements)
