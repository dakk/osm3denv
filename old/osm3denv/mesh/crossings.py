"""Zebra pedestrian crossings from ``highway=crossing`` nodes.

For every crossing node that sits on a road we emit a short stripe patch
across the carriageway: typically 6–8 white stripes 40 cm wide with 40 cm
gaps, spanning the road width. The stripes are drawn as flat quads sitting
just above the tarmac (large depth_bias in the material wins the depth
test).
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

# Road kinds we paint crossings on. Matches osm3denv.mesh.roads._classify.
_ROAD_WIDTHS = {
    "motorway": 12.0, "trunk": 10.0, "primary": 8.0, "secondary": 6.5,
    "tertiary": 5.5, "residential": 5.0, "unclassified": 5.0,
    "living_street": 4.0, "service": 3.0, "pedestrian": 4.0,
    "tertiary_link": 4.5, "secondary_link": 5.0, "primary_link": 6.0,
}

STRIPE_WIDTH = 0.45        # metres, along the crossing direction (across the road)
STRIPE_GAP   = 0.45        # gap between stripes
CROSSING_DEPTH = 4.0       # length of the crossing along the road direction
VERTICAL_OFFSET = 0.055    # just above the tarmac (sidewalk sits at 0.45)


@dataclass
class CrossingsMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    count: int


def _is_zebra(tags: dict[str, str]) -> bool:
    if tags.get("crossing:markings") == "zebra":
        return True
    if tags.get("crossing") == "zebra":
        return True
    if tags.get("crossing_ref") == "zebra":
        return True
    if tags.get("crossing:markings", "").startswith("zebra"):
        return True
    # Marked pedestrian crossings with unknown marking type — still default to
    # zebra as the visually dominant pattern in most European cities.
    return (tags.get("highway") == "crossing"
            and tags.get("crossing") in ("marked", "uncontrolled"))


def _find_parent_way(osm: OSMData, node_lon: float, node_lat: float):
    """Return (way, segment_index, tangent_unit) for the road whose geometry
    passes through the crossing node (1e-6 deg tolerance on either vertex).
    """
    best = None
    best_cost = 1e18
    tol2 = 1e-10
    for w in osm.ways:
        hw = w.tags.get("highway")
        if hw not in _ROAD_WIDTHS:
            continue
        if w.tags.get("tunnel") == "yes" or w.tags.get("bridge") == "yes":
            continue
        g = w.geometry
        for i, (lon, lat) in enumerate(g):
            dx = lon - node_lon; dy = lat - node_lat
            d2 = dx * dx + dy * dy
            if d2 < tol2 and d2 < best_cost:
                best = (w, i)
                best_cost = d2
    return best


def _tangent_at(way, idx: int, frame: Frame) -> tuple[float, float] | None:
    g = way.geometry
    n = len(g)
    if n < 2:
        return None
    i0 = max(0, idx - 1); i1 = min(n - 1, idx + 1)
    if i0 == i1:
        return None
    e0, n0 = frame.to_enu(g[i0][0], g[i0][1])
    e1, n1 = frame.to_enu(g[i1][0], g[i1][1])
    tx = float(e1) - float(e0)
    ty = float(n1) - float(n0)
    tl = (tx * tx + ty * ty) ** 0.5
    if tl < 1e-4:
        return None
    return (tx / tl, ty / tl)


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler) -> CrossingsMesh:
    verts: list[tuple[float, float, float]] = []
    norms: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    idx: list[tuple[int, int, int]] = []
    v_off = 0
    count = 0

    nodes = osm.filter_nodes(lambda t: t.get("highway") == "crossing")
    for node in nodes:
        if not _is_zebra(node.tags):
            continue
        hit = _find_parent_way(osm, node.lon, node.lat)
        if hit is None:
            continue
        way, seg_idx = hit
        tangent = _tangent_at(way, seg_idx, frame)
        if tangent is None:
            continue
        road_w = _ROAD_WIDTHS.get(way.tags.get("highway"), 5.0)

        # Crossing node in ENU.
        e, n = frame.to_enu(node.lon, node.lat)
        cx, cn = float(e), float(n)
        base_y = float(sampler.height_at(cx, cn)) + VERTICAL_OFFSET

        # Perpendicular (across the road) from the tangent (along the road).
        tx, ty = tangent
        perp = (-ty, tx)

        # Stripe layout: ``road_w`` spans across the road; stripes are
        # STRIPE_WIDTH across plus STRIPE_GAP between them, along the
        # crossing direction (= road perpendicular). Each stripe extends
        # CROSSING_DEPTH m along the road.
        half_depth = CROSSING_DEPTH * 0.5
        half_lat = road_w * 0.5
        pitch = STRIPE_WIDTH + STRIPE_GAP
        # Keep stripe count odd so one stripe is centred on the node.
        n_stripes = max(3, int(road_w / pitch) | 1)
        start = -(n_stripes * pitch - STRIPE_GAP) * 0.5
        for s in range(n_stripes):
            u0 = start + s * pitch
            u1 = u0 + STRIPE_WIDTH
            # Clip stripes so we don't paint past the carriageway.
            if u1 < -half_lat or u0 > half_lat:
                continue
            u0 = max(u0, -half_lat); u1 = min(u1, half_lat)
            if u1 - u0 < 0.05:
                continue
            for (ua, va, ub, vb) in [(u0, -half_depth, u1, +half_depth)]:
                # Four corners in the road-local frame → world ENU.
                a = (cx + ua * perp[0] + va * tx,
                     cn + ua * perp[1] + va * ty)
                b = (cx + ub * perp[0] + va * tx,
                     cn + ub * perp[1] + va * ty)
                c = (cx + ub * perp[0] + vb * tx,
                     cn + ub * perp[1] + vb * ty)
                d = (cx + ua * perp[0] + vb * tx,
                     cn + ua * perp[1] + vb * ty)
                # Convert to Ogre coords (east, y, -north).
                verts.extend([(a[0], base_y, -a[1]),
                              (b[0], base_y, -b[1]),
                              (c[0], base_y, -c[1]),
                              (d[0], base_y, -d[1])])
                norms.extend([(0.0, 1.0, 0.0)] * 4)
                uvs.extend([(0.0, 0.0), (1.0, 0.0),
                            (1.0, 1.0), (0.0, 1.0)])
                idx.extend([(v_off, v_off + 1, v_off + 2),
                            (v_off, v_off + 2, v_off + 3)])
                v_off += 4
        count += 1

    if count:
        log.info("crossings: %d zebra patches painted", count)
    if not verts:
        z3 = np.zeros((0, 3), dtype=np.float32)
        z2 = np.zeros((0, 2), dtype=np.float32)
        return CrossingsMesh(z3, z3,
                             np.zeros((0,), dtype=np.uint32), z2, 0)
    return CrossingsMesh(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(norms, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32).reshape(-1),
        uvs=np.asarray(uvs, dtype=np.float32),
        count=count,
    )
