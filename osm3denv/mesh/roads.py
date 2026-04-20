"""Road and railway ribbons draped on the terrain."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely.geometry as sg

from osm3denv.fetch.osm import OSMData, OSMWay
from osm3denv.frame import Frame
from osm3denv.mesh.buildings import parse_height
from osm3denv.mesh.drape import drape, planar_uv
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

_HIGHWAY_WIDTH = {
    "motorway": 12.0, "motorway_link": 8.0,
    "trunk": 10.0, "trunk_link": 7.0,
    "primary": 8.0, "primary_link": 6.0,
    "secondary": 6.0, "secondary_link": 5.0,
    "tertiary": 5.5, "tertiary_link": 4.5,
    "residential": 5.0, "unclassified": 5.0,
    "living_street": 4.0, "service": 3.0,
    "pedestrian": 4.0, "footway": 1.5, "path": 1.5, "cycleway": 2.0,
    "track": 3.0, "steps": 1.5,
}
_RAIL_WIDTH = 3.0


@dataclass
class RoadsMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    count: int


def _way_width(way: OSMWay) -> float:
    w = parse_height(way.tags.get("width"))
    if w is not None and w > 0:
        return w
    if "railway" in way.tags:
        return _RAIL_WIDTH
    hw = way.tags.get("highway")
    if hw in _HIGHWAY_WIDTH:
        return _HIGHWAY_WIDTH[hw]
    return 4.0


def _way_to_enu_line(way: OSMWay, frame: Frame) -> sg.LineString | None:
    lon = np.asarray([p[0] for p in way.geometry], dtype=np.float64)
    lat = np.asarray([p[1] for p in way.geometry], dtype=np.float64)
    east, north = frame.to_enu(lon, lat)
    coords = list(zip(east, north))
    if len(coords) < 2:
        return None
    line = sg.LineString(coords)
    return line if line.length > 0.5 else None


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler) -> RoadsMesh:
    all_v: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    all_i: list[np.ndarray] = []
    idx_offset = 0
    count = 0

    candidates = osm.filter_ways(lambda t: "highway" in t or "railway" in t)
    for way in candidates:
        if way.tags.get("tunnel") == "yes":
            continue  # hide tunnels; they live underground.
        line = _way_to_enu_line(way, frame)
        if line is None:
            continue
        width = _way_width(way)
        poly = line.buffer(width / 2.0, cap_style="flat", join_style="mitre")
        if poly.is_empty:
            continue

        is_bridge = way.tags.get("bridge") == "yes"
        if is_bridge:
            layer = parse_height(way.tags.get("layer")) or 1.0
            lift = max(3.0, 5.0 * float(layer))
            ys = [float(sampler.height_at(x, y)) for x, y in line.coords]
            flat_y = max(ys) + lift

        geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
        for g in geoms:
            if g.geom_type != "Polygon":
                continue
            if is_bridge:
                res = drape(g, sampler, per_vertex=False, flat_y=flat_y,
                            offset=0.0, max_step=1.5)
            else:
                res = drape(g, sampler, per_vertex=True, offset=0.40,
                            max_step=1.5)
            if res is None:
                continue
            v, n, i = res
            all_v.append(v)
            all_n.append(n)
            all_i.append(i + idx_offset)
            idx_offset += len(v)
            count += 1

    if not all_v:
        empty = np.zeros((0, 3), dtype=np.float32)
        empty_uv = np.zeros((0, 2), dtype=np.float32)
        return RoadsMesh(empty, empty, np.zeros((0,), dtype=np.uint32),
                         empty_uv, 0)
    vertices = np.concatenate(all_v, axis=0)
    return RoadsMesh(
        vertices=vertices,
        normals=np.concatenate(all_n, axis=0),
        indices=np.concatenate(all_i, axis=0),
        uvs=planar_uv(vertices, tile_m=1.0),  # meters; material scales for look
        count=count,
    )
