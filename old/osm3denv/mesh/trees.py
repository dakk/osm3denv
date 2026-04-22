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
import shapely.geometry as sg
import shapely.vectorized as sv
from shapely.ops import unary_union
from shapely.prepared import prep

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.geom import (parse_number, polygon_from_way,
                                polygons_from_relation)
from osm3denv.mesh.sample import TerrainSampler

# Approximate road widths used to keep the roadside scatter off the tarmac.
_ROADSIDE_WIDTHS = {
    "motorway": 12.0, "trunk": 10.0, "primary": 8.0, "secondary": 6.5,
    "tertiary": 5.5, "residential": 5.0, "unclassified": 5.0,
    "service": 3.0, "living_street": 4.0, "pedestrian": 4.0,
    "footway": 1.5, "path": 1.5, "cycleway": 2.0, "track": 3.0,
}
_ROADSIDE_SKIP = {"motorway", "motorway_link", "trunk", "trunk_link"}

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

    # --- Pass 3: roadside trees. ---------------------------------------
    # Sample each non-motorway road at 10 m intervals and drop a tree on
    # either side past the sidewalk, skipping anything that lands inside
    # a building footprint or another road polygon.
    remaining = MAX_TREES - len(placements)
    roadside_count = 0
    if remaining > 0:
        building_union = _building_union(osm, frame)
        road_union = _road_union(osm, frame)
        building_prep = prep(building_union) if building_union is not None else None
        road_prep = prep(road_union) if road_union is not None else None
        for way in osm.ways:
            if remaining <= 0:
                break
            hw = way.tags.get("highway")
            if hw is None or hw in _ROADSIDE_SKIP:
                continue
            if (way.tags.get("tunnel") == "yes"
                    or way.tags.get("bridge") == "yes"):
                continue
            lon = np.asarray([p[0] for p in way.geometry], dtype=np.float64)
            lat = np.asarray([p[1] for p in way.geometry], dtype=np.float64)
            if len(lon) < 2:
                continue
            east, north = frame.to_enu(lon, lat)
            coords = list(zip(east.tolist(), north.tolist()))
            try:
                line = sg.LineString(coords)
            except Exception:  # noqa: BLE001
                continue
            if line.length < 15.0:
                continue
            width = _ROADSIDE_WIDTHS.get(hw, 4.0)
            lat_off = width * 0.5 + 3.0   # past kerb+sidewalk
            step = 10.0
            d = step
            while d < line.length and remaining > 0:
                pt = line.interpolate(d)
                a = line.interpolate(max(d - 1.0, 0.0))
                b = line.interpolate(min(d + 1.0, line.length))
                tx = b.x - a.x; ty = b.y - a.y
                tl = (tx * tx + ty * ty) ** 0.5
                if tl < 1e-6:
                    d += step
                    continue
                perp = (-ty / tl, tx / tl)
                for sign in (+1.0, -1.0):
                    x = pt.x + sign * lat_off * perp[0]
                    y = pt.y + sign * lat_off * perp[1]
                    if radius_m is not None and (abs(x) > radius_m
                                                 or abs(y) > radius_m):
                        continue
                    candidate = sg.Point(x, y)
                    if building_prep is not None and building_prep.intersects(candidate):
                        continue
                    if road_prep is not None and road_prep.intersects(candidate):
                        continue
                    node_id = ((way.id * 100003 + int(d) * 10
                                + (1 if sign > 0 else 2)) & 0x7FFFFFFF)
                    h = 5.5 + (node_id % 6)
                    yaw = ((node_id * 2654435761) & 0xFFFF) / 0xFFFF * 2.0 * np.pi
                    placements.append(TreePlacement(
                        east=float(x), north=float(y),
                        base_y=float(sampler.height_at(x, y)),
                        height=float(h), yaw_rad=float(yaw),
                        species=_species({}, node_id),
                        seed=node_id,
                    ))
                    roadside_count += 1
                    remaining -= 1
                    if remaining <= 0:
                        break
                d += step
        if roadside_count:
            log.info("trees: %d roadside trees placed", roadside_count)

    return TreeRecords(placements=placements)


def _building_union(osm: OSMData, frame: Frame):
    polys = []
    for w in osm.filter_ways(lambda t: "building" in t or "building:part" in t):
        p = polygon_from_way(w, frame)
        if p is not None:
            polys.append(p)
    for r in osm.filter_relations(lambda t: "building" in t):
        for p in polygons_from_relation(r, frame):
            polys.append(p)
    if not polys:
        return None
    try:
        return unary_union(polys).buffer(0.8)
    except Exception:  # noqa: BLE001
        return None


def _road_union(osm: OSMData, frame: Frame):
    polys = []
    for w in osm.ways:
        hw = w.tags.get("highway")
        if hw is None:
            continue
        if w.tags.get("tunnel") == "yes":
            continue
        lon = np.asarray([p[0] for p in w.geometry], dtype=np.float64)
        lat = np.asarray([p[1] for p in w.geometry], dtype=np.float64)
        if len(lon) < 2:
            continue
        east, north = frame.to_enu(lon, lat)
        try:
            line = sg.LineString(list(zip(east.tolist(), north.tolist())))
        except Exception:  # noqa: BLE001
            continue
        if line.length <= 0.0:
            continue
        width = _ROADSIDE_WIDTHS.get(hw, 4.0)
        try:
            polys.append(line.buffer(width * 0.5 + 1.8,
                                     cap_style="flat", join_style="mitre"))
        except Exception:  # noqa: BLE001
            continue
    if not polys:
        return None
    try:
        return unary_union(polys)
    except Exception:  # noqa: BLE001
        return None
