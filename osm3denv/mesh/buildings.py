"""Extruded building meshes from OSM footprints.

Each building is deterministically bucketed into a material variant by its
way id (so a block gets a mix of brick/roof packs). Walls extrude to the
configured height with a flat top; a roof is then added on top whose shape
depends on the ``roof:shape`` OSM tag (or a sensible default inferred from
``building`` type):

* ``flat``       — the default; a planar roof polygon.
* ``pyramidal``  — all edges slope to a single apex above the centroid.
* ``gabled``     — two roof slopes meeting at a ridge along the long axis;
                   gable-end triangles fill in the short-side walls.
* ``hipped``     — four roof slopes meeting at an inset ridge along the long
                   axis; no extra wall geometry needed.

Gabled and hipped fall back to flat for non-rectangular footprints since
those shapes only have an unambiguous definition on a rectangular base.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely.geometry as sg
from mapbox_earcut import triangulate_float64

from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame
from osm3denv.mesh.geom import (
    parse_number as parse_height,
    polygon_from_relation,
    polygon_from_way,
)
from osm3denv.mesh.sample import TerrainSampler

log = logging.getLogger(__name__)

NUM_VARIANTS = 3

_ROOF_SHAPES = ("flat", "pyramidal", "gabled", "hipped")
_LEVEL_M = 3.0          # assumed storey height
_ROOF_LEVEL_M = 2.5     # shorter than a full storey


def _stable_hash(way_id: int) -> int:
    return (way_id * 2654435761) & 0xFFFFFFFF


def resolve_height(tags: dict[str, str], way_id: int = 0) -> float:
    h = parse_height(tags.get("height"))
    if h is not None and h > 0:
        return h
    levels = parse_height(tags.get("building:levels"))
    if levels is not None and levels > 0:
        return levels * _LEVEL_M
    jitter = (_stable_hash(way_id) & 0xFF) / 255.0
    return 6.0 + (jitter - 0.5) * 4.0


def _roof_shape(tags: dict[str, str], way_id: int, area_m2: float) -> str:
    """Return one of ``_ROOF_SHAPES``.

    Explicit OSM ``roof:shape`` wins if we support it; unknown shapes fall
    back to flat. Without a tag we pick based on the ``building`` type
    (residential → pitched, commercial → flat) with deterministic variation
    per way id so neighbouring houses don't all end up the same.
    """
    explicit = (tags.get("roof:shape") or "").lower()
    if explicit in _ROOF_SHAPES:
        return explicit
    if explicit:
        return "flat"
    b = (tags.get("building") or "yes").lower()
    if b in {"apartments", "commercial", "retail", "industrial", "office",
             "warehouse", "supermarket", "mall", "hangar"}:
        return "flat"
    if b in {"house", "detached", "semidetached_house", "terrace",
             "bungalow", "hut", "cabin", "farm", "barn"}:
        # Small pitched-roof residentials: split 50/50 between gabled/hipped
        return "gabled" if (_stable_hash(way_id) & 1) == 0 else "hipped"
    # Unknown: small footprints tend to be houses, large ones offices/shops.
    if area_m2 < 120.0:
        return "gabled" if (_stable_hash(way_id) & 1) == 0 else "hipped"
    return "flat"


def _roof_height(tags: dict[str, str], wall_height_m: float,
                 shape: str) -> float:
    if shape == "flat":
        return 0.0
    h = parse_height(tags.get("roof:height"))
    if h is not None and h > 0:
        return h
    levels = parse_height(tags.get("roof:levels"))
    if levels is not None and levels > 0:
        return levels * _ROOF_LEVEL_M
    # Defaults vary by shape: pyramidal is tallest (pointy), hipped is modest.
    default = {"gabled": 3.5, "hipped": 3.0, "pyramidal": 4.5}.get(shape, 0.0)
    # Clamp to 60% of wall height so we never produce a roof taller than walls.
    return min(default, wall_height_m * 0.6)


def _variant_for(way_id: int) -> int:
    return (_stable_hash(way_id) >> 8) % NUM_VARIANTS


@dataclass
class BuildingsMesh:
    variant: int
    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    uvs: np.ndarray
    count: int


# ---------------------------------------------------------------------------
# Rectangle detection + axes
# ---------------------------------------------------------------------------

def _is_near_rectangle(ring: np.ndarray, tol_deg: float = 18.0) -> bool:
    if len(ring) != 4:
        return False
    for i in range(4):
        a = ring[(i - 1) % 4]; b = ring[i]; c = ring[(i + 1) % 4]
        v1 = a - b; v2 = c - b
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        cos_a = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
        angle = np.degrees(np.arccos(cos_a))
        if abs(angle - 90.0) > tol_deg:
            return False
    return True


def _rect_axes(ring: np.ndarray):
    """Return (long_axis, short_axis, half_long, half_short, center) of a quad.

    Axes are unit vectors in (east, north). ``long_axis`` is the edge-aligned
    direction with the larger half-extent.
    """
    e01 = ring[1] - ring[0]
    e12 = ring[2] - ring[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))
    if l01 >= l12:
        long_axis = e01 / max(l01, 1e-6); half_long = l01 / 2.0
        half_short = l12 / 2.0
    else:
        long_axis = e12 / max(l12, 1e-6); half_long = l12 / 2.0
        half_short = l01 / 2.0
    short_axis = np.array([-long_axis[1], long_axis[0]])
    center = ring.mean(axis=0)
    return long_axis, short_axis, half_long, half_short, center


# ---------------------------------------------------------------------------
# Extruder
# ---------------------------------------------------------------------------

def _extrude(poly: sg.Polygon, wall_h: float, roof_h: float,
             shape: str, base_y: float):
    """Return (vertices, normals, uvs, indices) for one building, or None."""
    outer = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
    inners = [np.asarray(h.coords, dtype=np.float64)[:-1] for h in poly.interiors]
    rings = [outer] + inners
    flat2d = np.vstack(rings)
    ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    try:
        tri = triangulate_float64(flat2d, ring_ends).reshape(-1, 3)
    except Exception as exc:  # noqa: BLE001
        log.debug("earcut failed: %s", exc)
        return None

    wall_top_y = base_y + wall_h
    roof_top_y = wall_top_y + roof_h

    # Shared floor (fan of tri below the building, triangulated from earcut).
    n = len(flat2d)
    floor = np.stack([flat2d[:, 0], np.full(n, base_y),
                      -flat2d[:, 1]], axis=-1).astype(np.float32)
    floor_norms = np.tile([0.0, -1.0, 0.0], (n, 1)).astype(np.float32)
    floor_idx = (tri[:, [0, 2, 1]]).astype(np.uint32)
    floor_uvs = np.stack([flat2d[:, 0], flat2d[:, 1]],
                         axis=-1).astype(np.float32)

    # Per-shape roof can fall back to flat when we can't support the shape on
    # this particular footprint (e.g. non-rectangular gabled).
    use_shape = shape
    if use_shape in ("gabled", "hipped") and not _is_near_rectangle(outer):
        use_shape = "flat"
    if use_shape == "pyramidal" and len(outer) < 3:
        use_shape = "flat"

    # ----- Roof base (top of walls): always a flat ring at wall_top_y.
    # Earcut triangulation is reused for flat roofs.
    roof_base = np.stack([flat2d[:, 0], np.full(n, wall_top_y),
                          -flat2d[:, 1]], axis=-1).astype(np.float32)
    roof_base_uvs = np.stack([flat2d[:, 0], flat2d[:, 1]],
                             axis=-1).astype(np.float32)
    roof_base_norms = np.tile([0.0, 1.0, 0.0], (n, 1)).astype(np.float32)

    extra_verts: list[tuple[float, float, float]] = []
    extra_norms: list[tuple[float, float, float]] = []
    extra_uvs:   list[tuple[float, float]] = []
    extra_idx:   list[tuple[int, int, int]] = []

    # Walls are built from the rings with flat tops at wall_top_y.
    wall_verts: list[tuple[float, float, float]] = []
    wall_norms: list[tuple[float, float, float]] = []
    wall_uvs:   list[tuple[float, float]] = []
    wall_idx:   list[tuple[int, int, int]] = []
    WALL_TILE = 1.0
    top_v = wall_h / WALL_TILE

    for ring in rings:
        signed_area = 0.0
        m = len(ring)
        for i in range(m):
            signed_area += ring[i][0] * ring[(i + 1) % m][1] \
                         - ring[(i + 1) % m][0] * ring[i][1]
        if signed_area < 0:
            ring = ring[::-1]
        m = len(ring)
        u_running = 0.0
        for i in range(m):
            a = ring[i]; b = ring[(i + 1) % m]
            de = b[0] - a[0]; dn = b[1] - a[1]
            length = float(np.hypot(de, dn))
            if length < 1e-6:
                continue
            normal = (dn / length, 0.0, de / length)
            a_b = (float(a[0]), base_y, float(-a[1]))
            b_b = (float(b[0]), base_y, float(-b[1]))
            b_t = (float(b[0]), wall_top_y, float(-b[1]))
            a_t = (float(a[0]), wall_top_y, float(-a[1]))
            u0 = u_running / WALL_TILE
            u1 = (u_running + length) / WALL_TILE
            v0 = len(wall_verts)
            wall_verts.extend([a_b, b_b, b_t, a_t])
            wall_norms.extend([normal] * 4)
            wall_uvs.extend([(u0, 0.0), (u1, 0.0), (u1, top_v), (u0, top_v)])
            wall_idx.extend([(v0, v0 + 1, v0 + 2), (v0, v0 + 2, v0 + 3)])
            u_running += length

    # ----- Roof geometry per shape. -------------------------------------
    #
    # Each branch appends triangles to ``extra_*`` (or builds its own flat
    # roof using ``roof_base``). Everything is in absolute world coords.

    def _add_flat_roof():
        """Use the already-triangulated roof base as the flat roof."""
        nonlocal flat_roof_verts, flat_roof_norms, flat_roof_uvs, flat_roof_idx
        flat_roof_verts = roof_base
        flat_roof_norms = roof_base_norms
        flat_roof_uvs = roof_base_uvs
        flat_roof_idx = tri.astype(np.uint32)

    flat_roof_verts = np.zeros((0, 3), dtype=np.float32)
    flat_roof_norms = np.zeros((0, 3), dtype=np.float32)
    flat_roof_uvs = np.zeros((0, 2), dtype=np.float32)
    flat_roof_idx = np.zeros((0,), dtype=np.uint32)

    if use_shape == "flat":
        _add_flat_roof()

    elif use_shape == "pyramidal":
        # Apex above the centroid; fan-triangle from the apex to each outer
        # edge. Inner rings (holes) are treated as extra flat polygons using
        # the earcut base so holes don't get weird apex geometry.
        cx = float(outer[:, 0].mean()); cy = float(outer[:, 1].mean())
        apex = (cx, roof_top_y, -cy)

        # Outer ring CCW check: make sure normals point outward/up.
        ring = outer
        sa = 0.0
        mm = len(ring)
        for i in range(mm):
            sa += ring[i][0] * ring[(i+1) % mm][1] - ring[(i+1) % mm][0] * ring[i][1]
        if sa < 0:
            ring = ring[::-1]
        mm = len(ring)

        apex_idx = 0
        extra_verts.append(apex)
        extra_norms.append((0.0, 1.0, 0.0))  # placeholder, fixed below
        extra_uvs.append((cx, cy))

        for i in range(mm):
            a = ring[i]; b = ring[(i + 1) % mm]
            a_t = (float(a[0]), wall_top_y, float(-a[1]))
            b_t = (float(b[0]), wall_top_y, float(-b[1]))
            i_a = len(extra_verts)
            extra_verts.extend([a_t, b_t])
            # Face normal from apex/a/b
            va = np.array([a_t[0] - apex[0], a_t[1] - apex[1], a_t[2] - apex[2]])
            vb = np.array([b_t[0] - apex[0], b_t[1] - apex[1], b_t[2] - apex[2]])
            fn = np.cross(vb, va)
            ln = float(np.linalg.norm(fn))
            fn = fn / ln if ln > 1e-6 else np.array([0.0, 1.0, 0.0])
            extra_norms.extend([(float(fn[0]), float(fn[1]), float(fn[2]))] * 2)
            extra_uvs.extend([(a[0], a[1]), (b[0], b[1])])
            # Wind CCW when viewed from above: (a, b, apex)
            extra_idx.append((i_a, i_a + 1, apex_idx))

        # Inner holes: triangulate as flat caps at wall_top_y.
        if inners:
            # Build a flat roof JUST for holes: the earcut triangulation
            # already excludes the hole interior, so no hole-caps here.
            pass

    elif use_shape == "gabled":
        # Rectangle-only. Ridge along the long axis at roof_top_y, two roof
        # slopes, plus two gable-end triangles on the short-side walls.
        ring = outer.copy()
        sa = 0.0
        mm = len(ring)
        for i in range(mm):
            sa += ring[i][0] * ring[(i+1) % mm][1] - ring[(i+1) % mm][0] * ring[i][1]
        if sa < 0:
            ring = ring[::-1]
        long_axis, short_axis, half_long, half_short, center = _rect_axes(ring)

        # Ridge endpoints.
        r1 = center + long_axis * half_long
        r2 = center - long_axis * half_long

        def _to_world(p2d, y):
            return (float(p2d[0]), y, float(-p2d[1]))

        # Project each corner onto the long axis to tell which ridge end it
        # belongs to. Positive side = r1, negative side = r2.
        corners = [ring[i] for i in range(4)]
        sides = [float(np.dot(c - center, long_axis)) for c in corners]
        corners_pos = [c for c, s in zip(corners, sides) if s > 0]
        corners_neg = [c for c, s in zip(corners, sides) if s < 0]
        if len(corners_pos) != 2 or len(corners_neg) != 2:
            # Shouldn't happen for a proper quad; bail to flat.
            _add_flat_roof()
        else:
            # Each side of the ridge has two corners. Order them by
            # short-axis projection so the roof quads are wound correctly.
            def _order(pair, axis):
                return sorted(pair, key=lambda p: float(np.dot(p - center, axis)))
            p1_neg, p1_pos = _order(corners_pos, short_axis)   # r1 side
            p2_neg, p2_pos = _order(corners_neg, short_axis)   # r2 side

            # Each slope / gable triangle appends its own vertex copies below
            # so shading stays flat. Keep the ridge coords only as locals.
            r1_w = _to_world(r1, roof_top_y)
            r2_w = _to_world(r2, roof_top_y)

            def _add_slope(a, b, ridge_a, ridge_b, uv_a, uv_b,
                           uv_ra, uv_rb):
                """Add a roof-slope quad (triangles a→b→rb and a→rb→ra)."""
                base_i = len(extra_verts)
                v1 = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]])
                v2 = np.array([ridge_a[0] - a[0], ridge_a[1] - a[1],
                               ridge_a[2] - a[2]])
                fn = np.cross(v1, v2)
                ln = float(np.linalg.norm(fn))
                if ln > 1e-6:
                    fn = fn / ln
                else:
                    fn = np.array([0.0, 1.0, 0.0])
                n = (float(fn[0]), float(fn[1]), float(fn[2]))
                extra_verts.extend([a, b, ridge_b, ridge_a])
                extra_norms.extend([n] * 4)
                extra_uvs.extend([uv_a, uv_b, uv_rb, uv_ra])
                extra_idx.append((base_i, base_i + 1, base_i + 2))
                extra_idx.append((base_i, base_i + 2, base_i + 3))

            # Slope on the positive short-axis side (uses +half_short corners).
            a = _to_world(p1_pos, wall_top_y)
            b = _to_world(p2_pos, wall_top_y)
            ra = r1_w; rb = r2_w
            _add_slope(a, b, ra, rb,
                       (p1_pos[0], p1_pos[1]),
                       (p2_pos[0], p2_pos[1]),
                       (r1[0], r1[1]), (r2[0], r2[1]))

            # Slope on the negative short-axis side.
            a = _to_world(p2_neg, wall_top_y)
            b = _to_world(p1_neg, wall_top_y)
            _add_slope(a, b, r2_w, r1_w,
                       (p2_neg[0], p2_neg[1]),
                       (p1_neg[0], p1_neg[1]),
                       (r2[0], r2[1]), (r1[0], r1[1]))

            # Gable-end triangles (short-side walls going up to ridge).
            # Normal for each gable end: perpendicular to long_axis,
            # pointing away from centre. r1 end: +long_axis side.
            for ridge, ridge_uv, pos, neg, sign in [
                (r1_w, (r1[0], r1[1]), p1_pos, p1_neg, +1.0),
                (r2_w, (r2[0], r2[1]), p2_pos, p2_neg, -1.0),
            ]:
                a_t = _to_world(pos, wall_top_y)
                b_t = _to_world(neg, wall_top_y)
                normal_xy = long_axis * sign
                n = (float(normal_xy[0]), 0.0, float(normal_xy[1]))
                base_i = len(extra_verts)
                extra_verts.extend([a_t, b_t, ridge])
                extra_norms.extend([n, n, n])
                # UVs: continue the wall's "running length" pattern. Use
                # (projection onto short axis, height-as-v).
                extra_uvs.extend([
                    (float(np.dot(np.array(pos) - center, short_axis)), top_v),
                    (float(np.dot(np.array(neg) - center, short_axis)), top_v),
                    (0.0, (wall_h + roof_h) / WALL_TILE),
                ])
                # Winding for outward-facing normal: depends on sign.
                if sign > 0:
                    extra_idx.append((base_i, base_i + 1, base_i + 2))
                else:
                    extra_idx.append((base_i + 1, base_i, base_i + 2))

    elif use_shape == "hipped":
        # Four slopes to an inset ridge along the long axis. No extra walls
        # needed — wall tops stay flat. Rectangle-only.
        ring = outer.copy()
        sa = 0.0
        mm = len(ring)
        for i in range(mm):
            sa += ring[i][0] * ring[(i+1) % mm][1] - ring[(i+1) % mm][0] * ring[i][1]
        if sa < 0:
            ring = ring[::-1]
        long_axis, short_axis, half_long, half_short, center = _rect_axes(ring)
        # Inset the ridge by half_short along the long axis (classic hip).
        inset = min(half_short, half_long - 0.5)
        if inset <= 0.0:
            _add_flat_roof()
        else:
            r1 = center + long_axis * (half_long - inset)
            r2 = center - long_axis * (half_long - inset)

            corners = [ring[i] for i in range(4)]
            sides = [float(np.dot(c - center, long_axis)) for c in corners]
            corners_pos = [c for c, s in zip(corners, sides) if s > 0]
            corners_neg = [c for c, s in zip(corners, sides) if s < 0]

            def _order(pair, axis):
                return sorted(pair, key=lambda p: float(np.dot(p - center, axis)))

            p1_neg, p1_pos = _order(corners_pos, short_axis)
            p2_neg, p2_pos = _order(corners_neg, short_axis)

            def _to_world(p2d, y):
                return (float(p2d[0]), y, float(-p2d[1]))

            r1_w = _to_world(r1, roof_top_y)
            r2_w = _to_world(r2, roof_top_y)

            def _add_face(pts, uvs):
                """Add a flat-shaded polygon as a triangle fan."""
                base_i = len(extra_verts)
                v0, v1, v2 = pts[0], pts[1], pts[2]
                a = np.array([v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]])
                b = np.array([v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]])
                fn = np.cross(a, b)
                ln = float(np.linalg.norm(fn))
                fn = fn / ln if ln > 1e-6 else np.array([0.0, 1.0, 0.0])
                n = (float(fn[0]), float(fn[1]), float(fn[2]))
                for p, uv in zip(pts, uvs):
                    extra_verts.append(p)
                    extra_norms.append(n)
                    extra_uvs.append(uv)
                for i in range(1, len(pts) - 1):
                    extra_idx.append((base_i, base_i + i, base_i + i + 1))

            # Long-side trapezoids (two roof slopes running along the long axis).
            _add_face(
                [_to_world(p1_pos, wall_top_y), _to_world(p2_pos, wall_top_y),
                 r2_w, r1_w],
                [(p1_pos[0], p1_pos[1]), (p2_pos[0], p2_pos[1]),
                 (r2[0], r2[1]), (r1[0], r1[1])]
            )
            _add_face(
                [_to_world(p2_neg, wall_top_y), _to_world(p1_neg, wall_top_y),
                 r1_w, r2_w],
                [(p2_neg[0], p2_neg[1]), (p1_neg[0], p1_neg[1]),
                 (r1[0], r1[1]), (r2[0], r2[1])]
            )
            # Short-side triangles (hip end triangles).
            _add_face(
                [_to_world(p1_neg, wall_top_y), _to_world(p1_pos, wall_top_y), r1_w],
                [(p1_neg[0], p1_neg[1]), (p1_pos[0], p1_pos[1]), (r1[0], r1[1])]
            )
            _add_face(
                [_to_world(p2_pos, wall_top_y), _to_world(p2_neg, wall_top_y), r2_w],
                [(p2_pos[0], p2_pos[1]), (p2_neg[0], p2_neg[1]), (r2[0], r2[1])]
            )

    else:  # safety fallback
        _add_flat_roof()

    # ----- Assemble buffers. -------------------------------------------
    if wall_verts:
        wv = np.asarray(wall_verts, dtype=np.float32)
        wn = np.asarray(wall_norms, dtype=np.float32)
        wu = np.asarray(wall_uvs, dtype=np.float32)
        wi = np.asarray(wall_idx, dtype=np.uint32).reshape(-1)
    else:
        wv = np.zeros((0, 3), dtype=np.float32)
        wn = np.zeros((0, 3), dtype=np.float32)
        wu = np.zeros((0, 2), dtype=np.float32)
        wi = np.zeros((0,), dtype=np.uint32)

    if extra_verts:
        ev = np.asarray(extra_verts, dtype=np.float32)
        en = np.asarray(extra_norms, dtype=np.float32)
        eu = np.asarray(extra_uvs, dtype=np.float32)
        ei = np.asarray(extra_idx, dtype=np.uint32).reshape(-1)
    else:
        ev = np.zeros((0, 3), dtype=np.float32)
        en = np.zeros((0, 3), dtype=np.float32)
        eu = np.zeros((0, 2), dtype=np.float32)
        ei = np.zeros((0,), dtype=np.uint32)

    n_floor = len(floor)
    n_flat_roof = len(flat_roof_verts)
    n_wall = len(wv)

    # Offsets into the concatenated vertex array.
    floor_off = 0
    flat_roof_off = n_floor
    wall_off = flat_roof_off + n_flat_roof
    extra_off = wall_off + n_wall

    vertices = np.concatenate([floor, flat_roof_verts, wv, ev], axis=0)
    normals  = np.concatenate([floor_norms, flat_roof_norms, wn, en], axis=0)
    uvs      = np.concatenate([floor_uvs, flat_roof_uvs, wu, eu], axis=0)
    indices = np.concatenate([
        floor_idx.ravel() + floor_off,
        flat_roof_idx.ravel() + flat_roof_off,
        wi + wall_off,
        ei + extra_off,
    ], axis=0).astype(np.uint32)

    return vertices, normals, uvs, indices


def build(osm: OSMData, frame: Frame,
          sampler: TerrainSampler) -> list[BuildingsMesh]:
    """Build one BuildingsMesh per variant bucket."""
    buckets: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    def add(variant: int, tuple_):
        v, n, u, i = tuple_
        buckets.setdefault(variant, []).append((v, n, u, i))

    def _process(tags, way_id, poly):
        wall_h = resolve_height(tags, way_id)
        shape = _roof_shape(tags, way_id, float(poly.area))
        roof_h = _roof_height(tags, wall_h, shape)
        cx, cy = poly.centroid.x, poly.centroid.y
        base_y = float(sampler.height_at(cx, cy))
        return _extrude(poly, wall_h, roof_h, shape, base_y)

    for w in osm.filter_ways(lambda t: "building" in t or "building:part" in t):
        poly = polygon_from_way(w, frame)
        if poly is None:
            continue
        ext = _process(w.tags, w.id, poly)
        if ext is not None:
            add(_variant_for(w.id), ext)

    for r in osm.filter_relations(lambda t: "building" in t):
        poly = polygon_from_relation(r, frame)
        if poly is None:
            continue
        ext = _process(r.tags, r.id, poly)
        if ext is not None:
            add(_variant_for(r.id), ext)

    out: list[BuildingsMesh] = []
    for variant, parts in buckets.items():
        all_v = []; all_n = []; all_u = []; all_i = []
        off = 0
        count = 0
        for v, n, u, i in parts:
            all_v.append(v); all_n.append(n); all_u.append(u)
            all_i.append(i + off)
            off += len(v)
            count += 1
        out.append(BuildingsMesh(
            variant=variant,
            vertices=np.concatenate(all_v, axis=0),
            normals=np.concatenate(all_n, axis=0),
            indices=np.concatenate(all_i, axis=0),
            uvs=np.concatenate(all_u, axis=0),
            count=count,
        ))
    return out
