"""Street furniture: lamp posts and benches from OSM point nodes.

Generates cheap procedural meshes (prisms + boxes, ~30-50 tris per item) in
a single batched ManualObject per kind. Uses face normals (so each face is
its own 4-vertex quad) for flat shading — the simple shapes read cleaner
that way than smooth-shaded.
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

MAX_ITEMS = 2000


@dataclass
class FurnitureMesh:
    kind: str                # "lamp" or "bench"
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    count: int


# ---------------------------------------------------------------------------
# Primitive builders — each appends to output lists with a running vertex
# offset and returns the new offset. All use face-normal (flat shaded) quads
# so indices index into 4 fresh verts per face.
# ---------------------------------------------------------------------------

def _add_quad(verts, norms, uvs, idx, a, b, c, d, n, v_off: int) -> int:
    verts.extend([a, b, c, d])
    norms.extend([n, n, n, n])
    uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    idx.extend([(v_off, v_off + 1, v_off + 2),
                (v_off, v_off + 2, v_off + 3)])
    return v_off + 4


def _add_box(verts, norms, uvs, idx, cx: float, cy_base: float, cz: float,
             w: float, h: float, d: float, v_off: int,
             *, yaw: float = 0.0) -> int:
    """Axis-aligned box (optionally yawed around Y) with 6 flat-shaded faces."""
    hw, hd = w * 0.5, d * 0.5
    ya = y_base = cy_base
    yb = cy_base + h
    # Local corners in XZ (pre-yaw).
    p = [(-hw, -hd), (+hw, -hd), (+hw, +hd), (-hw, +hd)]
    if yaw != 0.0:
        c, s = np.cos(yaw), np.sin(yaw)
        p = [(x * c + z * s, -x * s + z * c) for (x, z) in p]
    # World corners.
    bot = [(cx + px, ya, cz + pz) for (px, pz) in p]
    top = [(cx + px, yb, cz + pz) for (px, pz) in p]

    # Face normals (rotate unit normals around Y by yaw too).
    def _rot(nx, nz):
        if yaw == 0.0:
            return (nx, 0.0, nz)
        c, s = np.cos(yaw), np.sin(yaw)
        return (nx * c + nz * s, 0.0, -nx * s + nz * c)

    # Top face (y=+1), bottom (y=-1), 4 side faces.
    v_off = _add_quad(verts, norms, uvs, idx,
                      top[0], top[1], top[2], top[3], (0.0, 1.0, 0.0), v_off)
    v_off = _add_quad(verts, norms, uvs, idx,
                      bot[0], bot[3], bot[2], bot[1], (0.0, -1.0, 0.0), v_off)
    for i in range(4):
        j = (i + 1) % 4
        # Outward normal for edge bot[i]→bot[j] on the XZ plane is the
        # rotated side unit vector. We derive it from the pre-yaw local
        # edge midpoint direction.
        mid_x = (p[i][0] + p[j][0]) * 0.5
        mid_z = (p[i][1] + p[j][1]) * 0.5
        nlen = (mid_x * mid_x + mid_z * mid_z) ** 0.5
        if nlen < 1e-6:
            nx, nz = 0.0, 0.0
        else:
            nx, nz = mid_x / nlen, mid_z / nlen
        n3 = (nx, 0.0, nz)
        # Note: we rotated p already so mid_x/mid_z are post-yaw; no double
        # rotation needed.
        v_off = _add_quad(verts, norms, uvs, idx,
                          bot[i], bot[j], top[j], top[i], n3, v_off)
    return v_off


# ---------------------------------------------------------------------------
# Item builders
# ---------------------------------------------------------------------------

def _build_lamp(cx: float, cy_base: float, cz: float,
                height: float, verts, norms, uvs, idx, v_off: int) -> int:
    """Simple lamp: thin vertical post + small box head."""
    pole_r = 0.07
    head_h = 0.30
    # Pole as an 0.14 × (height - head_h) × 0.14 box (reads fine at distance).
    v_off = _add_box(verts, norms, uvs, idx,
                     cx, cy_base, cz,
                     pole_r * 2, max(height - head_h, 0.5), pole_r * 2, v_off)
    # Head: slightly larger cube on top.
    head_w = 0.28
    v_off = _add_box(verts, norms, uvs, idx,
                     cx, cy_base + max(height - head_h, 0.5), cz,
                     head_w, head_h, head_w, v_off)
    return v_off


def _build_bench(cx: float, cy_base: float, cz: float, yaw: float,
                 verts, norms, uvs, idx, v_off: int) -> int:
    """Simple bench: seat slab + back slab + two side supports."""
    length = 1.5        # along the seat (pre-yaw, X axis)
    depth = 0.45        # perpendicular to length (Z axis)
    seat_h = 0.45       # seat top height above ground
    back_h = 0.40       # back height above seat
    slab_t = 0.05
    # Seat slab: (length, slab_t, depth)
    v_off = _add_box(verts, norms, uvs, idx,
                     cx, cy_base + seat_h - slab_t, cz,
                     length, slab_t, depth, v_off, yaw=yaw)
    # Side supports under the seat (0.05 × seat_h × depth), one at each end.
    c, s = np.cos(yaw), np.sin(yaw)
    for sign in (-1.0, +1.0):
        lx = sign * (length * 0.5 - 0.03)
        lz = 0.0
        wx = lx * c + lz * s
        wz = -lx * s + lz * c
        v_off = _add_box(verts, norms, uvs, idx,
                         cx + wx, cy_base, cz + wz,
                         0.06, seat_h - slab_t, depth, v_off, yaw=yaw)
    # Back slab: at the rear edge of the seat (+Z pre-yaw), length × back_h × slab_t
    back_z = -depth * 0.5 + slab_t * 0.5
    wx_b = 0.0 * c + back_z * s
    wz_b = -0.0 * s + back_z * c
    v_off = _add_box(verts, norms, uvs, idx,
                     cx + wx_b, cy_base + seat_h, cz + wz_b,
                     length, back_h, slab_t, v_off, yaw=yaw)
    return v_off


# ---------------------------------------------------------------------------
# Build entry point
# ---------------------------------------------------------------------------

def _empty(kind: str) -> FurnitureMesh:
    z3 = np.zeros((0, 3), dtype=np.float32)
    z2 = np.zeros((0, 2), dtype=np.float32)
    return FurnitureMesh(kind=kind, vertices=z3, normals=z3,
                         indices=np.zeros((0,), dtype=np.uint32), uvs=z2,
                         count=0)


def _finalise(kind: str, verts, norms, uvs, idx, count: int) -> FurnitureMesh:
    if not verts:
        return _empty(kind)
    return FurnitureMesh(
        kind=kind,
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(norms, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32).reshape(-1),
        uvs=np.asarray(uvs, dtype=np.float32),
        count=count,
    )


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler,
          *, radius_m: float | None = None) -> list[FurnitureMesh]:
    out: list[FurnitureMesh] = []

    # --- Lamp posts ---------------------------------------------------
    # Collect both explicit OSM street_lamp nodes AND synthetic lamps
    # auto-placed along major/minor roads, batched into one mesh.
    lamp_positions: list[tuple[float, float, float]] = []  # (east, north, height)
    lamp_nodes = osm.filter_nodes(lambda t: t.get("highway") == "street_lamp")
    for node in lamp_nodes[:MAX_ITEMS]:
        e, n = frame.to_enu(node.lon, node.lat)
        e = float(e); n = float(n)
        if radius_m is not None and (abs(e) > radius_m or abs(n) > radius_m):
            continue
        h = parse_number(node.tags.get("height"))
        if h is None or h <= 0.5 or h > 12.0:
            h = 4.0
        lamp_positions.append((e, n, float(h)))

    # Auto-scatter: every 30 m on each side of major/minor asphalt roads.
    # Offset laterally past the sidewalk strip (half_width + 1.7 m).
    _LAMP_ROAD_WIDTHS = {
        "motorway": 12.0, "trunk": 10.0, "primary": 8.0, "secondary": 6.5,
        "tertiary": 5.5, "residential": 5.0, "unclassified": 5.0,
        "living_street": 4.0, "service": 3.0,
    }
    _LAMP_MAX = 800
    auto_count = 0
    for way in osm.ways:
        if auto_count >= _LAMP_MAX:
            break
        hw = way.tags.get("highway")
        if hw not in _LAMP_ROAD_WIDTHS:
            continue
        if (way.tags.get("tunnel") == "yes"
                or way.tags.get("bridge") == "yes"):
            continue
        lon = np.asarray([p[0] for p in way.geometry], dtype=np.float64)
        lat = np.asarray([p[1] for p in way.geometry], dtype=np.float64)
        if len(lon) < 2:
            continue
        east, north = frame.to_enu(lon, lat)
        coords = np.stack([east, north], axis=-1)
        # Total length.
        segs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total = float(segs.sum())
        if total < 30.0:
            continue
        width = _LAMP_ROAD_WIDTHS[hw]
        lat_off = width * 0.5 + 1.7       # past the sidewalk
        step = 30.0
        # Precompute cumulative arc-length.
        cum = np.concatenate([[0.0], np.cumsum(segs)])
        d = 15.0   # first lamp offset from start
        while d < total and auto_count < _LAMP_MAX:
            # Interpolate (east, north) at arc-length d.
            idx = int(np.searchsorted(cum, d, side="right")) - 1
            idx = max(0, min(idx, len(segs) - 1))
            seg_len = segs[idx]
            if seg_len < 1e-6:
                d += step
                continue
            t = (d - cum[idx]) / seg_len
            p = coords[idx] + (coords[idx + 1] - coords[idx]) * t
            # Unit tangent for this segment; side alternates each step so
            # lamps zig-zag rather than facing each other.
            tan = (coords[idx + 1] - coords[idx]) / seg_len
            perp = np.array([-tan[1], tan[0]])
            sign = 1.0 if (int(d / step) & 1) == 0 else -1.0
            lamp_e = float(p[0] + sign * lat_off * perp[0])
            lamp_n = float(p[1] + sign * lat_off * perp[1])
            d += step
            if radius_m is not None and (abs(lamp_e) > radius_m
                                         or abs(lamp_n) > radius_m):
                continue
            lamp_positions.append((lamp_e, lamp_n, 5.0))
            auto_count += 1

    if lamp_positions:
        lv, ln, lu, li = [], [], [], []
        vertex_off = 0
        for e, n, h in lamp_positions:
            base_y = float(sampler.height_at(e, n))
            vertex_off = _build_lamp(e, base_y, -n, h, lv, ln, lu, li,
                                     vertex_off)
        out.append(_finalise("lamp", lv, ln, lu, li, len(lamp_positions)))
        log.info("furniture lamps: %d (%d osm + %d auto)",
                 len(lamp_positions),
                 len(lamp_positions) - auto_count, auto_count)

    # --- Benches ------------------------------------------------------
    bench_nodes = osm.filter_nodes(lambda t: t.get("amenity") == "bench")
    if bench_nodes:
        bv, bn, bu, bi = [], [], [], []
        off = 0; count = 0
        for node in bench_nodes[:MAX_ITEMS]:
            e, n = frame.to_enu(node.lon, node.lat)
            e = float(e); n = float(n)
            if radius_m is not None and (abs(e) > radius_m or abs(n) > radius_m):
                continue
            base_y = float(sampler.height_at(e, n))
            # Deterministic yaw per bench id so angles stay stable run-to-run.
            yaw = ((node.id * 2654435761) & 0xFFFF) / 0xFFFF * 2.0 * np.pi
            off = _build_bench(e, base_y, -n, float(yaw),
                               bv, bn, bu, bi, off)
            count += 1
        out.append(_finalise("bench", bv, bn, bu, bi, count))
        log.info("furniture benches: %d", count)

    return out
