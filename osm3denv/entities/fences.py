"""Fence / wall panels from OSM data, with per-strip LOD and texture variety.

Sources
-------
- ``landuse=residential`` polygon perimeters  (closed ring)
- Explicit ``barrier=fence|wall|retaining_wall`` ways  (open line)

Each strip is assigned one of three texture variants (concrete / brick /
plaster) deterministically from the OSM object id, so the same area always
looks the same between runs.  Every strip gets its own ``LODNode`` centred at
its polygon centroid so distance culling is per-strip rather than global.
"""
from __future__ import annotations

import logging

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

_FENCE_HEIGHT   = 1.8    # metres
_FENCE_Z_OFFSET = 0.05   # lift above terrain to avoid z-fighting
_FENCE_TEX_TILE = 2.0    # metres per texture repeat (horizontal)
_LOD_DIST       = 300.0  # metres — strip invisible beyond this

# Variant names must match keys in FENCE_ASSETS (textures.py)
_FENCE_VARIANTS = ["concrete", "brick", "plaster"]

# Fallback solid-colour RGB when a texture file is absent
_FALLBACK_COLOR: dict[str, bytes] = {
    "concrete": bytes([180, 175, 170]),
    "brick":    bytes([160,  85,  65]),
    "plaster":  bytes([228, 218, 198]),
}


# ---------------------------------------------------------------------------
# Terrain sampling
# ---------------------------------------------------------------------------

def _sample_z_vec(e_arr, n_arr, heightmap, grid, radius_m):
    scale = (grid - 1) / (2.0 * radius_m)
    col_f = np.clip((np.asarray(e_arr, np.float64) + radius_m) * scale, 0.0, grid - 1)
    row_f = np.clip((radius_m - np.asarray(n_arr, np.float64)) * scale, 0.0, grid - 1)
    r0 = np.minimum(row_f.astype(np.int32), grid - 2)
    c0 = np.minimum(col_f.astype(np.int32), grid - 2)
    fr = row_f - r0
    fc = col_f - c0
    return (
        heightmap[r0,   c0  ] * (1-fr)*(1-fc) +
        heightmap[r0,   c0+1] * (1-fr)*fc     +
        heightmap[r0+1, c0  ] * fr    *(1-fc) +
        heightmap[r0+1, c0+1] * fr    *fc
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Strip geometry builder
# ---------------------------------------------------------------------------

def _build_fence_strip(pts_e: np.ndarray, pts_n: np.ndarray,
                       z_vals: np.ndarray,
                       close_ring: bool = True,
                       offset_e: float = 0.0,
                       offset_n: float = 0.0) -> tuple | None:
    """Build centroid-relative fence geometry for one polyline.

    Vertices are stored as (pts_e - offset_e, pts_n - offset_n, z) so the
    caller can place a LODNode at (offset_e, offset_n, 0) and the geometry
    will land in the right world position.
    """
    if close_ring and (pts_e[0] != pts_e[-1] or pts_n[0] != pts_n[-1]):
        pts_e  = np.append(pts_e,  pts_e[0])
        pts_n  = np.append(pts_n,  pts_n[0])
        z_vals = np.append(z_vals, z_vals[0])

    n_segs = len(pts_e) - 1
    if n_segs < 1:
        return None

    verts   = np.empty((n_segs * 4, 3), np.float32)
    normals = np.empty((n_segs * 4, 3), np.float32)
    uvs     = np.empty((n_segs * 4, 2), np.float32)
    indices = np.empty((n_segs * 6,),   np.uint32)

    v_top = _FENCE_HEIGHT / _FENCE_TEX_TILE
    cum_u = 0.0

    for i in range(n_segs):
        e0 = float(pts_e[i])   - offset_e
        n0 = float(pts_n[i])   - offset_n
        e1 = float(pts_e[i+1]) - offset_e
        n1 = float(pts_n[i+1]) - offset_n
        z0 = float(z_vals[i])
        z1 = float(z_vals[i+1])

        dx, dy  = e1 - e0, n1 - n0
        seg_len = max(float(np.hypot(dx, dy)), 1e-6)
        tx, ty  = dx / seg_len, dy / seg_len
        nx, ny  = -ty, tx          # perpendicular (outward normal in XY)

        zb0, zb1 = z0 + _FENCE_Z_OFFSET, z1 + _FENCE_Z_OFFSET
        zt0, zt1 = zb0 + _FENCE_HEIGHT,  zb1 + _FENCE_HEIGHT

        u0 = cum_u / _FENCE_TEX_TILE
        u1 = (cum_u + seg_len) / _FENCE_TEX_TILE

        vi = i * 4
        verts[vi]   = [e0, n0, zb0]
        verts[vi+1] = [e1, n1, zb1]
        verts[vi+2] = [e1, n1, zt1]
        verts[vi+3] = [e0, n0, zt0]

        normals[vi:vi+4] = [nx, ny, 0.0]

        uvs[vi]   = [u0, 0.0]
        uvs[vi+1] = [u1, 0.0]
        uvs[vi+2] = [u1, v_top]
        uvs[vi+3] = [u0, v_top]

        ii = i * 6
        indices[ii:ii+6] = [vi, vi+1, vi+2, vi, vi+2, vi+3]

        cum_u += seg_len

    return verts, normals, uvs, indices


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Fences(MapEntity):
    """Fence panels with per-strip LOD and three texture variants."""

    _BARRIER_TAGS = {"fence", "wall", "retaining_wall"}

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, terrain,
                 fence_tex_paths: dict | None = None) -> None:
        self._osm             = osm
        self._frame           = frame
        self._radius_m        = radius_m
        self._terrain         = terrain
        self._fence_tex_paths = fence_tex_paths or {}
        # (verts_local, normals, uvs, indices, cx, cy, variant_idx)
        self._strips: list[tuple] = []

    def build(self) -> None:
        td        = self._terrain.data
        heightmap = td.heightmap
        grid      = heightmap.shape[0]
        r         = float(self._radius_m)

        seen: set[int] = set()
        n_residential = 0
        n_barrier     = 0

        def _process(pts, obj_id: int, *, close_ring: bool) -> bool:
            if len(pts) < 2 or obj_id in seen:
                return False
            lons = np.fromiter((p[0] for p in pts), np.float64, len(pts))
            lats = np.fromiter((p[1] for p in pts), np.float64, len(pts))
            east, north = self._frame.to_enu(lons, lats)
            if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                return False

            cx     = float(np.mean(east))
            cy     = float(np.mean(north))
            z_vals = _sample_z_vec(east, north, heightmap, grid, r)
            strip  = _build_fence_strip(east, north, z_vals,
                                        close_ring=close_ring,
                                        offset_e=cx, offset_n=cy)
            if strip is None:
                return False

            variant_idx = obj_id % len(_FENCE_VARIANTS)
            self._strips.append((*strip, cx, cy, variant_idx))
            seen.add(obj_id)
            return True

        for way in self._osm.filter_ways(lambda t: t.get("landuse") == "residential"):
            if _process(way.geometry, way.id, close_ring=True):
                n_residential += 1

        for rel in self._osm.filter_relations(lambda t: t.get("landuse") == "residential"):
            for role, ring in rel.rings:
                if role not in ("outer", "") or len(ring) < 4:
                    continue
                if _process(ring, rel.id, close_ring=True):
                    n_residential += 1
                    break

        for way in self._osm.filter_ways(
                lambda t: t.get("barrier") in self._BARRIER_TAGS):
            if _process(way.geometry, way.id, close_ring=False):
                n_barrier += 1

        log.info("fences: %d residential rings + %d barrier ways → %d strips",
                 n_residential, n_barrier, len(self._strips))

    def attach_to(self, parent) -> None:
        if not self._strips:
            return

        from osm3denv.render.helpers import attach_mesh, load_shader

        shader = load_shader("fence")

        def _tex(path, fallback_rgb, *, srgb):
            from panda3d.core import Filename, Texture
            tex = Texture()
            if path:
                try:
                    tex.read(Filename.fromOsSpecific(str(path)))
                    tex.setMinfilter(Texture.FT_linear_mipmap_linear)
                    tex.setMagfilter(Texture.FT_linear)
                    tex.setWrapU(Texture.WM_repeat)
                    tex.setWrapV(Texture.WM_repeat)
                    if srgb:
                        tex.setFormat(Texture.F_srgb)
                    return tex
                except Exception as exc:
                    log.warning("fence tex %s: %s", path, exc)
            tex.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb)
            tex.setRamImage(fallback_rgb)
            return tex

        # Group strips by variant and merge into one mesh per variant.
        # Vertices are stored centroid-relative; un-offset to world space before
        # concatenating so the parent node can sit at the origin.
        by_variant: dict[int, dict] = {}
        for verts, norms, uvs, indices, cx, cy, vidx in self._strips:
            if vidx not in by_variant:
                by_variant[vidx] = {"verts": [], "norms": [], "uvs": [],
                                     "indices": [], "v_count": 0}
            m = by_variant[vidx]
            v_world = verts.copy()
            v_world[:, 0] += cx
            v_world[:, 1] += cy
            m["verts"].append(v_world)
            m["norms"].append(norms)
            m["uvs"].append(uvs)
            m["indices"].append(indices + m["v_count"])
            m["v_count"] += len(verts)

        fence_root = parent.attachNewNode("fences")

        for vidx, m in by_variant.items():
            vname      = _FENCE_VARIANTS[vidx]
            variant_np = fence_root.attachNewNode(f"fence_{vname}")
            variant_np.setTwoSided(True)

            tex_p   = self._fence_tex_paths.get(vname, {})
            col_tex = _tex(tex_p.get("color"),  _FALLBACK_COLOR[vname], srgb=True)
            nrm_tex = _tex(tex_p.get("normal"), bytes([128, 128, 255]),  srgb=False)

            if shader:
                variant_np.setShader(shader)
                variant_np.setShaderInput("u_fence_col",     col_tex)
                variant_np.setShaderInput("u_fence_nrm",     nrm_tex)
                variant_np.setShaderInput("u_bump_strength", 0.6)

            all_verts = np.concatenate(m["verts"])
            all_norms = np.concatenate(m["norms"])
            all_uvs   = np.concatenate(m["uvs"])
            all_idx   = np.concatenate(m["indices"])
            attach_mesh(variant_np, "fgeom", all_verts, all_norms, all_uvs,
                        all_idx, depth_offset=1)

        log.info("fences: merged %d strips into %d draw calls",
                 len(self._strips), len(by_variant))
