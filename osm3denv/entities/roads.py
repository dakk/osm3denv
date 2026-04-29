"""Roads entity — 3D ribbon meshes for paved roads + splatmap for dirt tracks.

Paved roads (primary / secondary / tertiary / residential / service):
  3D ribbon geometry with ambientCG asphalt texture.
  Lane count → width:  primary/motorway/trunk = 4 lanes (~14 m),
                       secondary/tertiary      = 2 lanes (~ 7 m),
                       others                  = 1 lane  (~3.5 m).

Unpaved ways (track / path / footway / cycleway / …):
  Splatmap baked onto the terrain shader (same dirt-colour approach as before).
"""
from __future__ import annotations

import logging

import numpy as np

from osm3denv.entities.utils import sample_z_triangle
from osm3denv.entity import MapEntity
from osm3denv.fetch.osm import OSMData
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LANE_WIDTH_M    = 3.5   # metres per lane
_ROAD_Z_OFFSET   = 0.15  # metres above terrain
_ROAD_TEX_TILE_M = 3.5   # metres per texture repeat (one lane width)

# Spatial tiling for per-tile LOD
_ROAD_TILE_SIZE = 400.0   # metres per tile cell
_ROAD_LOD_NEAR  = 1000.0  # all roads visible within this distance from tile centre
_ROAD_LOD_FAR   = 4000.0  # only major roads visible from LOD_NEAR to LOD_FAR

# Road types kept at the medium LOD level (major only)
_MAJOR_HIGHWAY = {
    "motorway",  "motorway_link",
    "trunk",     "trunk_link",
    "primary",   "primary_link",
}

# Road types always rendered regardless of distance
_TRUNK_HIGHWAY = {"motorway", "motorway_link", "trunk", "trunk_link",
                  "primary",  "primary_link"}

# Paved roads → ribbon meshes
_RIBBON_LANES: dict[str, int] = {
    "motorway":       4,  "motorway_link":  2,
    "trunk":          4,  "trunk_link":     2,
    "primary":        4,  "primary_link":   2,
    "secondary":      2,  "secondary_link": 2,
    "tertiary":       2,  "tertiary_link":  1,
    "residential":    1,  "unclassified":   1,
    "service":        1,  "living_street":  1,
}

# Unpaved / pedestrian ways → terrain splatmap (dirt colour)
_SPLATMAP_HALF_W: dict[str, float] = {
    "track":      2.0,
    "path":       0.5,
    "footway":    0.5,
    "bridleway":  1.0,
    "cycleway":   0.6,
    "pedestrian": 1.5,
}

_SMAP_RES    = 2048
_MIN_HALF_PX = 0.8



# ---------------------------------------------------------------------------
# Splatmap rasteriser (for unpaved ways)
# ---------------------------------------------------------------------------

def _rasterize_segment(smap, x0, y0, x1, y1, half_w):
    H, W = smap.shape
    feather = max(1.5, half_w * 0.5)
    outer   = half_w + feather
    x_lo = max(0, int(min(x0, x1) - outer - 1))
    x_hi = min(W - 1, int(max(x0, x1) + outer + 1))
    y_lo = max(0, int(min(y0, y1) - outer - 1))
    y_hi = min(H - 1, int(max(y0, y1) + outer + 1))
    if x_hi < x_lo or y_hi < y_lo:
        return
    px = np.arange(x_lo, x_hi + 1, dtype=np.float32)
    py = np.arange(y_lo, y_hi + 1, dtype=np.float32)
    gx, gy = np.meshgrid(px, py)
    dx, dy = x1 - x0, y1 - y0
    seg_len_sq = float(dx*dx + dy*dy)
    if seg_len_sq > 1e-12:
        t = np.clip(((gx - x0)*dx + (gy - y0)*dy) / seg_len_sq, 0.0, 1.0)
        cx = x0 + t*dx;  cy = y0 + t*dy
    else:
        cx, cy = float(x0), float(y0)
    dist  = np.sqrt((gx - cx)**2 + (gy - cy)**2)
    inner = half_w - feather
    s     = np.clip((dist - inner) / (2.0 * feather), 0.0, 1.0)
    val   = 1.0 - s*s*(3.0 - 2.0*s)
    np.maximum(smap[y_lo:y_hi+1, x_lo:x_hi+1], val,
               out=smap[y_lo:y_hi+1, x_lo:x_hi+1])


# ---------------------------------------------------------------------------
# Ribbon mesh builder (for paved roads)
# ---------------------------------------------------------------------------

def _build_ribbon(pts_e, pts_n, pts_z, half_w):
    """Return (vertices, normals, uvs, indices) with CCW (upward-facing) winding."""
    n = len(pts_e)
    if n < 2:
        return None

    seg_dx  = np.diff(pts_e)
    seg_dy  = np.diff(pts_n)
    seg_dz  = np.diff(pts_z)
    seg_len = np.hypot(seg_dx, seg_dy).clip(1e-6)
    seg_tx  = seg_dx / seg_len
    seg_ty  = seg_dy / seg_len
    seg_tz  = seg_dz / seg_len

    # Per-node tangents (average adjacent segments)
    tan_x = np.empty(n, np.float32)
    tan_y = np.empty(n, np.float32)
    tan_z = np.empty(n, np.float32)
    tan_x[0]  = seg_tx[0];   tan_y[0]  = seg_ty[0];   tan_z[0]  = seg_tz[0]
    tan_x[-1] = seg_tx[-1];  tan_y[-1] = seg_ty[-1];  tan_z[-1] = seg_tz[-1]
    for i in range(1, n - 1):
        tx = seg_tx[i-1] + seg_tx[i]
        ty = seg_ty[i-1] + seg_ty[i]
        tz = seg_tz[i-1] + seg_tz[i]
        l  = max(float(np.hypot(tx, ty)), 1e-6)
        tan_x[i] = tx / l;  tan_y[i] = ty / l;  tan_z[i] = tz / l

    # Right-hand perpendicular in XY plane
    perp_x = -tan_y
    perp_y =  tan_x

    # Left (south-of-travel) and right (north-of-travel) edge positions
    lx = pts_e - perp_x * half_w;  ly = pts_n - perp_y * half_w
    rx = pts_e + perp_x * half_w;  ry = pts_n + perp_y * half_w

    # Surface normal: cross(tangent_3d, right_direction_3d)
    # tangent=(tx,ty,tz), right=(perp_x,perp_y,0)
    # N = (ty*0-tz*perp_y,  tz*perp_x-tx*0,  tx*perp_y-ty*perp_x)
    nm_x = -tan_z * perp_y
    nm_y =  tan_z * perp_x
    nm_z =  tan_x * tan_x + tan_y * tan_y   # ≈1 on flat terrain
    nm_l = np.sqrt(nm_x**2 + nm_y**2 + nm_z**2).clip(1e-6)
    nm_x /= nm_l;  nm_y /= nm_l;  nm_z /= nm_l

    # UV: U tiles lane-count times across width, V advances along length
    cum_dist = np.zeros(n, np.float32)
    cum_dist[1:] = np.cumsum(seg_len).astype(np.float32)
    v_coords = cum_dist / _ROAD_TEX_TILE_M
    u_right  = (half_w * 2.0) / _ROAD_TEX_TILE_M

    z_col = pts_z + _ROAD_Z_OFFSET

    verts = np.empty((n * 2, 3), np.float32)
    norms = np.empty((n * 2, 3), np.float32)
    uvs   = np.empty((n * 2, 2), np.float32)

    verts[0::2, 0] = lx;    verts[0::2, 1] = ly;    verts[0::2, 2] = z_col
    verts[1::2, 0] = rx;    verts[1::2, 1] = ry;    verts[1::2, 2] = z_col
    norms[0::2] = norms[1::2] = np.stack([nm_x, nm_y, nm_z], axis=1)
    uvs[0::2, 0] = 0.0;       uvs[0::2, 1] = v_coords
    uvs[1::2, 0] = u_right;   uvs[1::2, 1] = v_coords

    # CCW quads (front face points up / +Z):
    #   left[i], left[i+1], right[i+1]   and   left[i], right[i+1], right[i]
    seg_n = n - 1
    row   = np.arange(seg_n, dtype=np.uint32)
    li    = row * 2;        ri  = row * 2 + 1
    li1   = (row + 1) * 2;  ri1 = (row + 1) * 2 + 1
    indices = np.stack([li, li1, ri1, li, ri1, ri], axis=1).ravel()

    return verts, norms, uvs, indices


def _merge_ribbons(ribbons):
    all_v, all_n, all_uv, all_idx = [], [], [], []
    base = 0
    for res in ribbons:
        if res is None:
            continue
        v, n, uv, idx = res
        all_v.append(v);    all_n.append(n)
        all_uv.append(uv);  all_idx.append(idx + base)
        base += len(v)
    if not all_v:
        return None
    return (np.concatenate(all_v),  np.concatenate(all_n),
            np.concatenate(all_uv), np.concatenate(all_idx))


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class Roads(MapEntity):
    """Paved-road ribbon meshes + unpaved-track splatmap."""

    def __init__(self, osm: OSMData, frame: Frame,
                 radius_m: float, terrain,
                 road_tex_paths: dict | None = None) -> None:
        self._osm            = osm
        self._frame          = frame
        self._radius_m       = radius_m
        self._terrain        = terrain
        self._road_tex_paths = road_tex_paths or {}
        # {(ti, tj): {"full": [ribbon, ...], "major": [ribbon, ...]}}
        self._road_tiles: dict[tuple, dict] = {}
        self._splatmap: np.ndarray | None = None

    def build(self) -> None:
        td        = self._terrain.data
        heightmap = td.heightmap
        grid      = heightmap.shape[0]
        r         = float(self._radius_m)

        # --- Ribbon meshes for paved roads, bucketed into spatial tiles ---
        n_ribbon = 0
        for way in self._osm.filter_ways(lambda t: t.get("highway") in _RIBBON_LANES):
            geom = way.geometry
            if len(geom) < 2:
                continue
            lons = np.fromiter((p[0] for p in geom), np.float64, len(geom))
            lats = np.fromiter((p[1] for p in geom), np.float64, len(geom))
            e, n = self._frame.to_enu(lons, lats)
            if e.max() < -r or e.min() > r or n.max() < -r or n.min() > r:
                continue
            hwy    = way.tags["highway"]
            lanes  = _RIBBON_LANES[hwy]
            half_w = lanes * _LANE_WIDTH_M / 2.0
            z      = sample_z_triangle(e, n, heightmap, grid, r)
            ribbon = _build_ribbon(e, n, z, half_w)
            if ribbon is None:
                continue
            # Assign to tile based on the way midpoint
            mid = len(e) // 2
            ti  = int(float(e[mid]) // _ROAD_TILE_SIZE)
            tj  = int(float(n[mid]) // _ROAD_TILE_SIZE)
            if (ti, tj) not in self._road_tiles:
                self._road_tiles[(ti, tj)] = {"full": [], "major": [], "trunk": []}
            self._road_tiles[(ti, tj)]["full"].append(ribbon)
            if hwy in _MAJOR_HIGHWAY:
                self._road_tiles[(ti, tj)]["major"].append(ribbon)
            if hwy in _TRUNK_HIGHWAY:
                self._road_tiles[(ti, tj)]["trunk"].append(ribbon)
            n_ribbon += 1

        # --- Splatmap for dirt tracks / footways ---
        smap   = np.zeros((_SMAP_RES, _SMAP_RES), dtype=np.float32)
        n_smap = 0
        for way in self._osm.filter_ways(lambda t: t.get("highway") in _SPLATMAP_HALF_W):
            geom = way.geometry
            if len(geom) < 2:
                continue
            lons = np.fromiter((p[0] for p in geom), np.float64, len(geom))
            lats = np.fromiter((p[1] for p in geom), np.float64, len(geom))
            east, north = self._frame.to_enu(lons, lats)
            if east.max() < -r or east.min() > r or north.max() < -r or north.min() > r:
                continue
            px      = (east  + r) / (2.0 * r) * _SMAP_RES
            py      = (north + r) / (2.0 * r) * _SMAP_RES
            hw_m    = _SPLATMAP_HALF_W[way.tags["highway"]]
            hw_px   = max(_MIN_HALF_PX, hw_m / (2.0 * r) * _SMAP_RES)
            for i in range(len(px) - 1):
                _rasterize_segment(smap, px[i], py[i], px[i+1], py[i+1], hw_px)
            n_smap += 1

        try:
            from scipy.ndimage import gaussian_filter
            smap = gaussian_filter(smap, sigma=0.8)
        except ImportError:
            pass
        self._splatmap = smap.clip(0.0, 1.0)

        log.info("roads: %d ribbon ways, %d splatmap ways", n_ribbon, n_smap)

    def attach_to(self, parent) -> None:
        from osm3denv.render.helpers import attach_mesh, load_shader

        # --- Splatmap → terrain shader input ---
        if self._splatmap is not None:
            from panda3d.core import Texture
            data = (self._splatmap * 255.0).clip(0, 255).astype(np.uint8)
            tex  = Texture("road_splatmap")
            res  = self._splatmap.shape[0]
            tex.setup2dTexture(res, res, Texture.T_unsigned_byte, Texture.F_luminance)
            tex.setRamImage(memoryview(data))
            terrain_np = parent.find("**/terrain")
            target = terrain_np if not terrain_np.isEmpty() else parent
            target.setShaderInput("u_road_splatmap", tex)

        # --- Ribbon meshes → per-tile LOD nodes ---
        if not self._road_tiles:
            return

        shader = load_shader("road")

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
                    log.warning("road tex %s: %s", path, exc)
            tex.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb)
            tex.setRamImage(fallback_rgb)
            return tex

        road_p   = self._road_tex_paths.get("road", {})
        road_col = _tex(road_p.get("color"),  bytes([55, 55, 60]),    srgb=True)
        road_nrm = _tex(road_p.get("normal"), bytes([128, 128, 255]), srgb=False)

        from panda3d.core import LODNode
        road_root = parent.attachNewNode("roads")
        road_root.setTwoSided(True)
        if shader:
            road_root.setShader(shader)
            road_root.setShaderInput("u_road_col",      road_col)
            road_root.setShaderInput("u_road_nrm",      road_nrm)
            road_root.setShaderInput("u_bump_strength", 1.5)

        n_tiles = 0
        for (ti, tj), tile in self._road_tiles.items():
            full_geom  = _merge_ribbons(tile["full"])
            if full_geom is None:
                continue

            # Tile centre in world space — LODNode sits here so distance
            # is measured to the centre of this tile's road content.
            cx = (ti + 0.5) * _ROAD_TILE_SIZE
            cn = (tj + 0.5) * _ROAD_TILE_SIZE

            lod_np = road_root.attachNewNode(LODNode(f"rt_{ti}_{tj}"))
            lod_np.setPos(float(cx), float(cn), 0.0)
            lod = lod_np.node()

            def _attach_tile_mesh(parent_np, geom, label):
                v, n, uv, idx = geom
                v = v.copy(); v[:, 0] -= cx; v[:, 1] -= cn  # tile-relative
                attach_mesh(parent_np, label, v, n, uv, idx, depth_offset=1)

            # Child 0: all roads (near)
            full_np = lod_np.attachNewNode("full")
            _attach_tile_mesh(full_np, full_geom, "rgeom_f")
            lod.addSwitch(_ROAD_LOD_NEAR, 0.0)

            # Child 1: major roads only (medium distance)
            major_np = lod_np.attachNewNode("major")
            major_geom = _merge_ribbons(tile["major"])
            if major_geom is not None:
                _attach_tile_mesh(major_np, major_geom, "rgeom_m")
            lod.addSwitch(_ROAD_LOD_FAR, _ROAD_LOD_NEAR)

            # Child 2: trunk/motorway — always visible beyond LOD_FAR
            trunk_np = lod_np.attachNewNode("trunk")
            trunk_geom = _merge_ribbons(tile["trunk"])
            if trunk_geom is not None:
                _attach_tile_mesh(trunk_np, trunk_geom, "rgeom_t")
            lod.addSwitch(1e9, _ROAD_LOD_FAR)

            n_tiles += 1

        log.info("roads: %d tiles, LOD near=%.0fm far=%.0fm",
                 n_tiles, _ROAD_LOD_NEAR, _ROAD_LOD_FAR)
