"""Vegetation + landuse area polygons draped on terrain.

Classifies each OSM area by `leisure` / `natural` / `landuse` tag (in that
precedence order) and groups polygons by the chosen material. One ManualObject
per material — each AreaMesh holds one material's geometry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.drape import drape
from osm3denv.mesh.geom import polygon_from_way, polygons_from_relation
from osm3denv.mesh.sample import TerrainSampler
from osm3denv.render import materials

log = logging.getLogger(__name__)


# Precedence: first matching entry wins, so leisure > natural > landuse.
# (tag_key, value_set, material_factory, y_offset_m)
AREA_CATEGORIES: list[tuple[str, set[str] | None, object, float]] = [
    ("leisure", {"park", "garden", "nature_reserve", "village_green",
                 "pitch", "playground", "recreation_ground", "common"},
                materials.vegetation, 0.10),
    ("natural", {"wood", "scrub", "grassland", "heath"},
                materials.vegetation, 0.10),
    ("natural", {"bare_rock", "scree"},
                materials.rock, 0.08),
    ("natural", {"sand", "beach"},
                materials.sand, 0.08),
    ("landuse", {"forest", "meadow", "grass", "cemetery",
                 "orchard", "vineyard", "recreation_ground",
                 "village_green", "allotments"},
                materials.vegetation, 0.10),
    ("landuse", {"farmland", "farmyard"},
                materials.farmland, 0.06),
    ("landuse", {"residential"},
                materials.residential, 0.05),
    ("landuse", {"commercial", "retail"},
                materials.commercial, 0.05),
    ("landuse", {"industrial", "construction", "railway"},
                materials.industrial, 0.05),
]


@dataclass
class AreaMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    material_factory: object  # callable() -> str, resolved at render time
    count: int


def _classify(tags: dict[str, str]):
    """Return (material_factory, offset_m) or None for this set of tags."""
    for key, values, mat, offset in AREA_CATEGORIES:
        v = tags.get(key)
        if v is None:
            continue
        if values is None or v in values:
            return mat, offset
    return None


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler) -> list[AreaMesh]:
    # Bucket polygons by material factory (don't call the factory yet —
    # Ogre's MaterialManager isn't available until the render context starts).
    buckets: dict[object, tuple[float, list]] = {}

    def _queue(poly, mat_factory, offset: float) -> None:
        if mat_factory not in buckets:
            buckets[mat_factory] = (offset, [])
        buckets[mat_factory][1].append(poly)

    for way in osm.ways:
        cls = _classify(way.tags)
        if cls is None:
            continue
        mat_factory, offset = cls
        poly = polygon_from_way(way, frame)
        if poly is not None:
            _queue(poly, mat_factory, offset)

    for rel in osm.relations:
        cls = _classify(rel.tags)
        if cls is None:
            continue
        mat_factory, offset = cls
        for poly in polygons_from_relation(rel, frame):
            _queue(poly, mat_factory, offset)

    results: list[AreaMesh] = []
    for mat_factory, (offset, polys) in buckets.items():
        all_v: list[np.ndarray] = []
        all_n: list[np.ndarray] = []
        all_i: list[np.ndarray] = []
        idx_off = 0
        count = 0
        for poly in polys:
            geoms = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
            for g in geoms:
                if g.geom_type != "Polygon":
                    continue
                res = drape(g, sampler, per_vertex=True, offset=offset, max_step=5.0)
                if res is None:
                    continue
                v, n, i = res
                all_v.append(v)
                all_n.append(n)
                all_i.append(i + idx_off)
                idx_off += len(v)
                count += 1
        if not all_v:
            continue
        results.append(AreaMesh(
            vertices=np.concatenate(all_v, axis=0),
            normals=np.concatenate(all_n, axis=0),
            indices=np.concatenate(all_i, axis=0),
            material_factory=mat_factory,
            count=count,
        ))
    return results
