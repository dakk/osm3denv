"""Obelisks and columns from ``man_made=obelisk|column``.

These features are commonly tagged both as ways (with building:part=yes + a
height tag) and as stand-alone nodes. A plain building-style extrusion reads
as a bland prism, which is very wrong for the Roman skyline — this module
substitutes proper tapered obelisks and cylindrical columns.

The buildings module filters out ``man_made=obelisk|column`` so we don't
double-render.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.geom import parse_number, polygon_from_way
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

CYL_SEGMENTS = 20


@dataclass
class MonumentsMesh:
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    count: int


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _append_cylinder(cx: float, cz: float, y_base: float,
                     radius: float, height: float,
                     verts, norms, uvs, idx, v_off: int) -> int:
    """Capped smooth cylinder (side + top + bottom)."""
    n = CYL_SEGMENTS
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos_a = np.cos(angles); sin_a = np.sin(angles)

    # Side — unique pair of verts per segment with outward normal.
    for i in range(n):
        j = (i + 1) % n
        xa = cx + radius * cos_a[i]; za = cz + radius * sin_a[i]
        xb = cx + radius * cos_a[j]; zb = cz + radius * sin_a[j]
        # Average of the two corner outward normals (makes facetted look).
        mx = (cos_a[i] + cos_a[j]) * 0.5
        mz = (sin_a[i] + sin_a[j]) * 0.5
        ml = (mx * mx + mz * mz) ** 0.5 or 1.0
        nml = (mx / ml, 0.0, mz / ml)
        verts.extend([(xa, y_base, za), (xb, y_base, zb),
                      (xb, y_base + height, zb), (xa, y_base + height, za)])
        norms.extend([nml] * 4)
        u0 = i / n; u1 = (i + 1) / n
        uvs.extend([(u0, 0.0), (u1, 0.0), (u1, height), (u0, height)])
        idx.extend([(v_off, v_off + 1, v_off + 2),
                    (v_off, v_off + 2, v_off + 3)])
        v_off += 4

    # Top cap — triangle fan around centre.
    centre_top = v_off
    verts.append((cx, y_base + height, cz))
    norms.append((0.0, 1.0, 0.0))
    uvs.append((0.5, 0.5))
    v_off += 1
    ring_start = v_off
    for i in range(n):
        verts.append((cx + radius * cos_a[i], y_base + height,
                      cz + radius * sin_a[i]))
        norms.append((0.0, 1.0, 0.0))
        uvs.append(((cos_a[i] + 1.0) * 0.5, (sin_a[i] + 1.0) * 0.5))
    v_off += n
    for i in range(n):
        idx.append((centre_top, ring_start + i,
                    ring_start + (i + 1) % n))

    # Bottom cap.
    centre_bot = v_off
    verts.append((cx, y_base, cz))
    norms.append((0.0, -1.0, 0.0))
    uvs.append((0.5, 0.5))
    v_off += 1
    ring_start_b = v_off
    for i in range(n):
        verts.append((cx + radius * cos_a[i], y_base,
                      cz + radius * sin_a[i]))
        norms.append((0.0, -1.0, 0.0))
        uvs.append(((cos_a[i] + 1.0) * 0.5, (sin_a[i] + 1.0) * 0.5))
    v_off += n
    for i in range(n):
        idx.append((centre_bot, ring_start_b + (i + 1) % n,
                    ring_start_b + i))
    return v_off


def _append_obelisk(cx: float, cz: float, y_base: float,
                    base_half: float, height: float,
                    verts, norms, uvs, idx, v_off: int,
                    *, yaw: float = 0.0) -> int:
    """Four-sided tapered prism with a small pyramid cap.

    * Plinth (the bottom 6% of height, full ``base_half`` size, not tapered).
    * Shaft (tapers linearly from ``base_half`` at plinth top to ~0.55 × at
      90% of height).
    * Pyramidion (pyramid cap from 90% to 100% of height).
    """
    plinth_h = max(0.3, height * 0.06)
    pyr_start = y_base + height * 0.90
    shaft_top_half = base_half * 0.55
    pyr_top_y = y_base + height

    c, s = np.cos(yaw), np.sin(yaw)

    def xy(half_x: float, half_z: float, sign_x: float, sign_z: float):
        lx = sign_x * half_x
        lz = sign_z * half_z
        return (cx + lx * c + lz * s, -lx * s + lz * c + cz)

    # ---- Plinth (4 side quads + bottom) ------------------------------
    y_a = y_base
    y_b = y_base + plinth_h
    corners_bot = [xy(base_half, base_half, sx, sz)
                   for (sx, sz) in ((-1, -1), (+1, -1), (+1, +1), (-1, +1))]
    corners_top = corners_bot  # plinth has no taper

    for i in range(4):
        j = (i + 1) % 4
        ax, az = corners_bot[i]
        bx, bz = corners_bot[j]
        # Outward normal.
        dx, dz = bx - ax, bz - az
        dl = (dx * dx + dz * dz) ** 0.5 or 1.0
        nx, nz = dz / dl, -dx / dl
        if nx * ((ax + bx) * 0.5 - cx) + nz * ((az + bz) * 0.5 - cz) < 0:
            nx, nz = -nx, -nz
        verts.extend([(ax, y_a, az), (bx, y_a, bz),
                      (bx, y_b, bz), (ax, y_b, az)])
        norms.extend([(nx, 0.0, nz)] * 4)
        uvs.extend([(0.0, 0.0), (1.0, 0.0),
                    (1.0, plinth_h), (0.0, plinth_h)])
        idx.extend([(v_off, v_off + 1, v_off + 2),
                    (v_off, v_off + 2, v_off + 3)])
        v_off += 4

    # Bottom face (underside, hidden usually).
    verts.extend([(corners_bot[0][0], y_a, corners_bot[0][1]),
                  (corners_bot[3][0], y_a, corners_bot[3][1]),
                  (corners_bot[2][0], y_a, corners_bot[2][1]),
                  (corners_bot[1][0], y_a, corners_bot[1][1])])
    norms.extend([(0.0, -1.0, 0.0)] * 4)
    uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    idx.extend([(v_off, v_off + 1, v_off + 2),
                (v_off, v_off + 2, v_off + 3)])
    v_off += 4

    # ---- Shaft (tapered from plinth top to pyramidion base) ----------
    y_a = y_base + plinth_h
    y_b = pyr_start
    sh_bot = [xy(base_half, base_half, sx, sz)
              for (sx, sz) in ((-1, -1), (+1, -1), (+1, +1), (-1, +1))]
    sh_top = [xy(shaft_top_half, shaft_top_half, sx, sz)
              for (sx, sz) in ((-1, -1), (+1, -1), (+1, +1), (-1, +1))]
    for i in range(4):
        j = (i + 1) % 4
        ax, az = sh_bot[i]; bx, bz = sh_bot[j]
        tx, tz = sh_top[i]; ux, uz = sh_top[j]
        dx, dz = bx - ax, bz - az
        dl = (dx * dx + dz * dz) ** 0.5 or 1.0
        nx, nz = dz / dl, -dx / dl
        if nx * ((ax + bx) * 0.5 - cx) + nz * ((az + bz) * 0.5 - cz) < 0:
            nx, nz = -nx, -nz
        # Add slight upward tilt because of taper.
        ny = (base_half - shaft_top_half) / max(y_b - y_a, 1e-3)
        nl = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
        nml = (nx / nl, ny / nl, nz / nl)
        verts.extend([(ax, y_a, az), (bx, y_a, bz),
                      (ux, y_b, uz), (tx, y_b, tz)])
        norms.extend([nml] * 4)
        uvs.extend([(0.0, 0.0), (1.0, 0.0),
                    (1.0, y_b - y_a), (0.0, y_b - y_a)])
        idx.extend([(v_off, v_off + 1, v_off + 2),
                    (v_off, v_off + 2, v_off + 3)])
        v_off += 4

    # ---- Pyramidion cap ----------------------------------------------
    apex = (cx, pyr_top_y, cz)
    base_corners = sh_top
    for i in range(4):
        j = (i + 1) % 4
        ax, az = base_corners[i]
        bx, bz = base_corners[j]
        # Face normal from triangle (a, apex, b).
        v1 = (apex[0] - ax, apex[1] - y_b, apex[2] - az)
        v2 = (bx - ax, 0.0, bz - az)
        nx = v1[1] * v2[2] - v1[2] * v2[1]
        ny = v1[2] * v2[0] - v1[0] * v2[2]
        nz = v1[0] * v2[1] - v1[1] * v2[0]
        nl = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
        nml = (nx / nl, ny / nl, nz / nl)
        verts.extend([(ax, y_b, az), (bx, y_b, bz), apex])
        norms.extend([nml] * 3)
        uvs.extend([(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
        idx.append((v_off, v_off + 1, v_off + 2))
        v_off += 3
    return v_off


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _empty() -> MonumentsMesh:
    z3 = np.zeros((0, 3), dtype=np.float32)
    z2 = np.zeros((0, 2), dtype=np.float32)
    return MonumentsMesh(z3, z3, np.zeros((0,), dtype=np.uint32), z2, 0)


def _height_for(tags: dict[str, str], kind: str) -> float:
    h = parse_number(tags.get("height"))
    if h is None or h <= 0.5:
        # Fallback typical sizes.
        h = 18.0 if kind == "obelisk" else 12.0
    return float(min(h, 80.0))


def build(osm: OSMData, frame: Frame, sampler: TerrainSampler) -> MonumentsMesh:
    verts, norms, uvs, idx = [], [], [], []
    v_off = 0
    count = 0

    # Polygon/way features.
    way_candidates = osm.filter_ways(
        lambda t: t.get("man_made") in ("obelisk", "column"))
    for way in way_candidates:
        kind = way.tags["man_made"]
        poly = polygon_from_way(way, frame, min_area=0.25)
        if poly is None:
            continue
        cx, cz_enu = float(poly.centroid.x), float(poly.centroid.y)
        base_y = float(sampler.height_at(cx, cz_enu))
        h = _height_for(way.tags, kind)
        # Characteristic size from the footprint (use sqrt(area) so shape is
        # orientation-independent).
        ch = max(0.5, np.sqrt(float(poly.area)))
        if kind == "column":
            radius = max(0.35, min(2.5, ch * 0.45))
            v_off = _append_cylinder(cx, -cz_enu, base_y, radius, h,
                                     verts, norms, uvs, idx, v_off)
        else:
            base_half = max(0.5, min(2.5, ch * 0.5))
            # Use minrotated rectangle orientation so the obelisk's square
            # base aligns with the footprint when it's rectangular.
            try:
                rect = poly.minimum_rotated_rectangle
                rc = list(rect.exterior.coords)
                if len(rc) >= 2:
                    yaw = float(np.arctan2(rc[1][1] - rc[0][1],
                                            rc[1][0] - rc[0][0]))
                else:
                    yaw = 0.0
            except Exception:  # noqa: BLE001
                yaw = 0.0
            v_off = _append_obelisk(cx, -cz_enu, base_y, base_half, h,
                                    verts, norms, uvs, idx, v_off, yaw=yaw)
        count += 1

    # Point/node features.
    node_candidates = osm.filter_nodes(
        lambda t: t.get("man_made") in ("obelisk", "column"))
    for node in node_candidates:
        kind = node.tags["man_made"]
        e, n = frame.to_enu(node.lon, node.lat)
        cx, cz = float(e), -float(n)
        base_y = float(sampler.height_at(float(e), float(n)))
        h = _height_for(node.tags, kind)
        if kind == "column":
            v_off = _append_cylinder(cx, cz, base_y, 0.5, h,
                                     verts, norms, uvs, idx, v_off)
        else:
            v_off = _append_obelisk(cx, cz, base_y, 0.8, h,
                                    verts, norms, uvs, idx, v_off)
        count += 1

    if count == 0:
        return _empty()
    log.info("monuments: %d (obelisk/column)", count)
    return MonumentsMesh(
        vertices=np.asarray(verts, dtype=np.float32),
        normals=np.asarray(norms, dtype=np.float32),
        indices=np.asarray(idx, dtype=np.uint32).reshape(-1),
        uvs=np.asarray(uvs, dtype=np.float32),
        count=count,
    )
