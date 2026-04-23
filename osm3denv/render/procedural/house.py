"""Procedural residential house geometry generator for Panda3D.

Each house is built at three LOD levels:
  full   — walls, roof, windows, frames, door, balcony, chimney, AC
  medium — walls + roof only  (no per-pixel details)
  simple — single textured box

Shared textures (wall, roof) are pre-loaded once in HouseBuilder.__init__
and reused across every building so only O(1) Texture objects are created
per material type, not O(n_buildings).
"""
from __future__ import annotations

import math
from random import Random

import numpy as np

_FLAT_NRM = bytes([128, 128, 255])


# ── geometry primitives ───────────────────────────────────────────────────────

def _quad(p0, p1, p2, p3, normal, u_max: float, v_max: float):
    n = np.asarray(normal, np.float32)
    ln = float(np.linalg.norm(n))
    if ln > 0:
        n = n / ln
    v = np.array([p0, p1, p2, p3], np.float32)
    ns = np.tile(n, (4, 1))
    u = np.array([[0,0],[u_max,0],[u_max,v_max],[0,v_max]], np.float32)
    i = np.array([0,1,2, 0,2,3], np.uint32)
    return v, ns, u, i


def _tri(p0, p1, p2, normal, uv0, uv1, uv2):
    n = np.asarray(normal, np.float32)
    ln = float(np.linalg.norm(n))
    if ln > 0:
        n = n / ln
    v  = np.array([p0, p1, p2], np.float32)
    ns = np.tile(n, (3, 1))
    u  = np.array([uv0, uv1, uv2], np.float32)
    i  = np.array([0, 1, 2], np.uint32)
    return v, ns, u, i


def _merge(*parts):
    Vs, Ns, Us, Is = [], [], [], []
    off = 0
    for v, n, u, idx in parts:
        Vs.append(v); Ns.append(n); Us.append(u); Is.append(idx + off)
        off += len(v)
    return (np.concatenate(Vs), np.concatenate(Ns),
            np.concatenate(Us), np.concatenate(Is))


def _box(x0, x1, y0, y1, z0, z1, ts: float = 2.0):
    dx, dy, dz = (x1-x0)/ts, (y1-y0)/ts, (z1-z0)/ts
    return _merge(
        _quad((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1), (0,-1,0), dx, dz),
        _quad((x1,y1,z0),(x0,y1,z0),(x0,y1,z1),(x1,y1,z1), (0, 1,0), dx, dz),
        _quad((x0,y1,z0),(x0,y0,z0),(x0,y0,z1),(x0,y1,z1), (-1,0,0), dy, dz),
        _quad((x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1), ( 1,0,0), dy, dz),
        _quad((x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1), (0, 0,1), dx, dy),
        _quad((x1,y0,z0),(x0,y0,z0),(x0,y1,z0),(x1,y1,z0), (0, 0,-1),dx, dy),
    )


def _attach(parent, name: str, geo, shader, inputs: dict):
    verts, norms, uvs, idxs = geo
    if len(verts) == 0:
        return None
    from osm3denv.render.helpers import attach_mesh
    np_ = attach_mesh(parent, name, verts, norms, uvs, idxs)
    if shader:
        np_.setShader(shader)
        for k, v in inputs.items():
            np_.setShaderInput(k, v)
    return np_


def _load_tex(path, fallback_rgb: bytes, *, srgb: bool):
    from panda3d.core import Filename, Texture
    tex = Texture()
    if path is not None:
        try:
            tex.read(Filename.fromOsSpecific(str(path)))
            tex.setMinfilter(Texture.FT_linear_mipmap_linear)
            tex.setMagfilter(Texture.FT_linear)
            tex.setWrapU(Texture.WM_repeat)
            tex.setWrapV(Texture.WM_repeat)
            if srgb:
                tex.setFormat(Texture.F_srgb)
            return tex
        except Exception:
            pass
    tex.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb)
    tex.setRamImage(fallback_rgb)
    return tex


# ── HouseBuilder ──────────────────────────────────────────────────────────────

class HouseBuilder:
    """Generates Panda3D NodePaths for residential buildings at three LOD levels.

    Textures are pre-loaded once in ``__init__`` and shared across every
    ``build_*`` call, so the texture RAM cost is O(material_types) not
    O(n_buildings).
    """

    FLOOR_H  = 3.0
    PITCH    = 0.55
    OVERHANG = 0.45
    TEX_WALL = 2.0
    TEX_ROOF = 1.5
    WIN_W    = 0.90
    WIN_H    = 1.10
    WIN_SILL = 0.90
    WIN_RC   = 0.06
    DOOR_W   = 0.95
    DOOR_H   = 2.05
    FT       = 0.08

    def __init__(self, tex_paths: dict, shader=None) -> None:
        self._sh          = shader
        self._solid_cache: dict = {}
        from osm3denv.render.helpers import load_shader
        self._glass_sh    = load_shader("building_glass")

        tp = tex_paths
        # Brick kept separately for chimneys (always brick)
        self._brick_col = _load_tex(tp.get("brick", {}).get("color"),  bytes([170, 90, 62]), srgb=True)
        self._brick_nrm = _load_tex(tp.get("brick", {}).get("normal"), _FLAT_NRM,            srgb=False)

        # Wall material pool: (col_tex, nrm_tex) — picked randomly per house.
        # Brick appears only once so it is uncommon (~12 %).
        _wall_specs = [
            ("brick",             bytes([170,  90,  62])),   # red-orange clay brick
            ("plaster",           bytes([225, 215, 198])),   # cream plaster
            ("plaster_b",         bytes([205, 195, 178])),   # warm off-white plaster
            ("plaster_c",         bytes([188, 196, 185])),   # cool-tinted plaster
            ("painted_plaster",   bytes([210, 175, 140])),   # painted plaster A
            ("painted_plaster_b", bytes([160, 185, 170])),   # painted plaster B
            ("painted_plaster_c", bytes([190, 170, 155])),   # painted plaster C
            ("concrete",          bytes([158, 154, 148])),   # exposed concrete
        ]
        self._wall_mats: list[tuple] = [
            (_load_tex(tp.get(k, {}).get("color"),  fb,       srgb=True),
             _load_tex(tp.get(k, {}).get("normal"), _FLAT_NRM, srgb=False))
            for k, fb in _wall_specs
        ]

        self._roof_col = _load_tex(tp.get("roof_tiles", {}).get("color"),  bytes([128, 52, 32]), srgb=True)
        self._roof_nrm = _load_tex(tp.get("roof_tiles", {}).get("normal"), _FLAT_NRM,            srgb=False)

        # Shared solid-colour textures
        self._glass_tex = self._s(38, 52, 72)
        self._flat_nrm  = self._s(128, 128, 255)

    # ── LOD entry points ─────────────────────────────────────────────────────

    def build_full(self, seed: int, width: float, depth: float,
                   floors: int, parent) -> object:
        """All features: walls, roof, windows, frames, door, balcony, chimney, AC."""
        rng = Random(seed)
        wall_col, wall_nrm = self._wall_tex(rng)
        root = parent.attachNewNode(f"hf_{seed}")
        hw = width / 2.0; hd = depth / 2.0; h = floors * self.FLOOR_H

        self._walls(root, hw, hd, h, wall_col, wall_nrm)
        self._roof(root, hw, hd, h)

        n_win   = max(1, int(hw * 2 / 2.6))
        spacing = (hw * 2 - n_win * self.WIN_W) / (n_win + 1)
        self._windows(root, rng, hw, hd, floors, n_win, spacing)
        self._door(root, rng, hd)

        if floors >= 2 and rng.random() < 0.65:
            self._balcony(root, rng, hw, hd, floors, wall_col, wall_nrm)
        if rng.random() < 0.55:
            self._chimney(root, rng, hw, hd, h)
        if rng.random() < 0.45:
            self._ac(root, rng, hw, hd, floors)
        return root

    def build_medium(self, seed: int, width: float, depth: float,
                     floors: int, parent) -> object:
        """Walls + roof only — no small details."""
        rng = Random(seed)
        wall_col, wall_nrm = self._wall_tex(rng)
        root = parent.attachNewNode(f"hm_{seed}")
        hw = width / 2.0; hd = depth / 2.0; h = floors * self.FLOOR_H
        self._walls(root, hw, hd, h, wall_col, wall_nrm)
        self._roof(root, hw, hd, h)
        return root

    def build_simple(self, seed: int, width: float, depth: float,
                     floors: int, parent) -> object:
        """Single textured box — one draw call per building."""
        rng = Random(seed)
        wall_col, wall_nrm = self._wall_tex(rng)
        root = parent.attachNewNode(f"hs_{seed}")
        hw = width / 2.0; hd = depth / 2.0; h = floors * self.FLOOR_H
        sh = self._sh
        _attach(root, "box", _box(-hw, hw, -hd, hd, 0.0, h, self.TEX_WALL),
                sh, {"u_col_tex": wall_col, "u_nrm_tex": wall_nrm,
                     "u_bump_strength": 0.0})
        return root

    # ── geometry builders ────────────────────────────────────────────────────

    def _walls(self, root, hw, hd, h, wall_col, wall_nrm):
        sh = self._sh; ts = self.TEX_WALL
        _attach(root, "walls", _merge(
            _quad((-hw,-hd,0),(hw,-hd,0),(hw,-hd,h),(-hw,-hd,h),   (0,-1,0), hw*2/ts, h/ts),
            _quad((hw,hd,0),(-hw,hd,0),(-hw,hd,h),(hw,hd,h),       (0, 1,0), hw*2/ts, h/ts),
            _quad((-hw,hd,0),(-hw,-hd,0),(-hw,-hd,h),(-hw,hd,h),   (-1,0,0), hd*2/ts, h/ts),
            _quad((hw,-hd,0),(hw,hd,0),(hw,hd,h),(hw,-hd,h),        ( 1,0,0), hd*2/ts, h/ts),
        ), sh, {"u_col_tex": wall_col, "u_nrm_tex": wall_nrm, "u_bump_strength": 1.5})

    def _roof(self, root, hw, hd, h):
        sh = self._sh; ov = self.OVERHANG; tr = self.TEX_ROOF
        rh   = h + self.PITCH * hd
        rise = rh - h; run = hd + ov
        slp  = math.hypot(rise, run)
        W    = hw + ov
        roof_np = _attach(root, "roof", _merge(
            # Slopes (CCW winding — outward normals verified)
            _quad((-W,-hd-ov,h),(W,-hd-ov,h),(W,0,rh),(-W,0,rh),
                  (0,-rise,run),   W*2/tr, slp/tr),
            _quad((W,hd+ov,h),(-W,hd+ov,h),(-W,0,rh),(W,0,rh),
                  (0, rise,run),   W*2/tr, slp/tr),
            # Gable ends
            _tri((-W,hd+ov,h),(-W,-hd-ov,h),(-W,0,rh), (-1,0,0),
                 [0,0],[(hd+ov)*2/tr,0],[(hd+ov)/tr,rise/tr]),
            _tri((W,-hd-ov,h),(W,hd+ov,h),(W,0,rh),   ( 1,0,0),
                 [0,0],[(hd+ov)*2/tr,0],[(hd+ov)/tr,rise/tr]),
            # Soffits — underside of overhang
            _quad((-W,-hd-ov,h),(-hw,-hd,h),(hw,-hd,h),(W,-hd-ov,h), (0,0,-1), W*2/tr, ov/tr),
            _quad((W, hd+ov,h),(hw, hd,h),(-hw,hd,h),(-W, hd+ov,h), (0,0,-1), W*2/tr, ov/tr),
        ), sh, {"u_col_tex": self._roof_col, "u_nrm_tex": self._roof_nrm,
                "u_bump_strength": 1.2})
        if roof_np:
            roof_np.setTwoSided(True)

    def _windows(self, root, rng, hw, hd, floors, n_win, spacing):
        sh = self._sh; ft = self.FT
        glass_parts, frame_parts = [], []

        # Front / back walls (Y-facing)
        for face_y, ny in [(-hd, -1), (hd, 1)]:
            for fl in range(floors):
                z0 = fl * self.FLOOR_H + self.WIN_SILL
                z1 = z0 + self.WIN_H
                for wi in range(n_win):
                    cx  = -hw + spacing*(wi+1) + self.WIN_W*wi + self.WIN_W*0.5
                    wx0 = cx - self.WIN_W*0.5; wx1 = cx + self.WIN_W*0.5
                    gy  = face_y + ny*self.WIN_RC
                    glass_parts.append(
                        _quad((wx0,gy,z0),(wx1,gy,z0),(wx1,gy,z1),(wx0,gy,z1),
                              (0,ny,0), 1.0, 1.0))
                    fy = face_y + ny*(self.WIN_RC - 0.01)
                    for fp in [
                        ((wx0-ft,fy,z0-ft),(wx1+ft,fy,z0-ft),(wx1+ft,fy,z0),(wx0-ft,fy,z0)),
                        ((wx0-ft,fy,z1),(wx1+ft,fy,z1),(wx1+ft,fy,z1+ft),(wx0-ft,fy,z1+ft)),
                        ((wx0-ft,fy,z0-ft),(wx0,fy,z0-ft),(wx0,fy,z1+ft),(wx0-ft,fy,z1+ft)),
                        ((wx1,fy,z0-ft),(wx1+ft,fy,z0-ft),(wx1+ft,fy,z1+ft),(wx1,fy,z1+ft)),
                    ]:
                        frame_parts.append(_quad(*fp, (0,ny,0), 1.0, 1.0))

        # Side walls (X-facing) — number of windows fitted along depth
        n_win_x   = max(1, int(hd * 2 / 2.6))
        spacing_x = (hd * 2 - n_win_x * self.WIN_W) / (n_win_x + 1)
        for face_x, nx in [(-hw, -1), (hw, 1)]:
            for fl in range(floors):
                z0 = fl * self.FLOOR_H + self.WIN_SILL
                z1 = z0 + self.WIN_H
                for wi in range(n_win_x):
                    cy  = -hd + spacing_x*(wi+1) + self.WIN_W*wi + self.WIN_W*0.5
                    wy0 = cy - self.WIN_W*0.5; wy1 = cy + self.WIN_W*0.5
                    gx  = face_x + nx*self.WIN_RC
                    glass_parts.append(
                        _quad((gx,wy0,z0),(gx,wy1,z0),(gx,wy1,z1),(gx,wy0,z1),
                              (nx,0,0), 1.0, 1.0))
                    fx = face_x + nx*(self.WIN_RC - 0.01)
                    for fp in [
                        ((fx,wy0-ft,z0-ft),(fx,wy1+ft,z0-ft),(fx,wy1+ft,z0),(fx,wy0-ft,z0)),
                        ((fx,wy0-ft,z1),(fx,wy1+ft,z1),(fx,wy1+ft,z1+ft),(fx,wy0-ft,z1+ft)),
                        ((fx,wy0-ft,z0-ft),(fx,wy0,z0-ft),(fx,wy0,z1+ft),(fx,wy0-ft,z1+ft)),
                        ((fx,wy1,z0-ft),(fx,wy1+ft,z0-ft),(fx,wy1+ft,z1+ft),(fx,wy1,z1+ft)),
                    ]:
                        frame_parts.append(_quad(*fp, (nx,0,0), 1.0, 1.0))

        trim_rgb = rng.choice([(240,237,230),(250,245,235),(225,222,215)])
        trim_tex = self._s(*trim_rgb)
        fi_glass = {"u_col_tex": self._glass_tex, "u_nrm_tex": self._flat_nrm,
                    "u_bump_strength": 0.0, "u_specular": 1.0}
        fi_trim  = {"u_col_tex": trim_tex, "u_nrm_tex": self._flat_nrm,
                    "u_bump_strength": 0.0, "u_specular": 0.0}
        if glass_parts:
            glass_np = _attach(root, "glass", _merge(*glass_parts), sh, fi_glass)
            if glass_np:
                glass_np.setTwoSided(True)
        if frame_parts:
            _attach(root, "frames", _merge(*frame_parts), sh, fi_trim)

    def _door(self, root, rng, hd):
        sh = self._sh; ft = self.FT
        dx0 = -self.DOOR_W*0.5; dx1 = self.DOOR_W*0.5; dy = -hd-0.03
        door_rgb = rng.choice([(72,42,18),(52,32,14),(35,55,85),(28,50,28),(100,30,20)])
        door_tex = self._s(*door_rgb)
        trim_tex = self._s(240, 237, 230)
        fi_door = {"u_col_tex": door_tex, "u_nrm_tex": self._flat_nrm, "u_bump_strength": 0.0}
        fi_trim = {"u_col_tex": trim_tex, "u_nrm_tex": self._flat_nrm, "u_bump_strength": 0.0}
        _attach(root, "door",
                _quad((dx0,dy,0),(dx1,dy,0),(dx1,dy,self.DOOR_H),(dx0,dy,self.DOOR_H),
                      (0,-1,0), 1, 1),
                sh, fi_door)
        _attach(root, "door_trim", _merge(
            _quad((dx0-ft,dy-.01,0),(dx0,dy-.01,0),(dx0,dy-.01,self.DOOR_H+ft),(dx0-ft,dy-.01,self.DOOR_H+ft),(0,-1,0),1,1),
            _quad((dx1,dy-.01,0),(dx1+ft,dy-.01,0),(dx1+ft,dy-.01,self.DOOR_H+ft),(dx1,dy-.01,self.DOOR_H+ft),(0,-1,0),1,1),
            _quad((dx0-ft,dy-.01,self.DOOR_H),(dx1+ft,dy-.01,self.DOOR_H),(dx1+ft,dy-.01,self.DOOR_H+ft),(dx0-ft,dy-.01,self.DOOR_H+ft),(0,-1,0),1,1),
        ), sh, fi_trim)

    def _balcony(self, root, rng, hw, hd, floors, wall_col, wall_nrm):
        sh = self._sh
        fl  = rng.randint(1, floors-1)
        bw  = min(hw*2*rng.uniform(0.40, 0.72), 3.6)
        bd  = rng.uniform(1.0, 1.8)
        bx0 = -bw*0.5; bx1 = bw*0.5; bz = fl * self.FLOOR_H
        trim = self._s(240, 237, 230)
        fi_wall = {"u_col_tex": wall_col, "u_nrm_tex": wall_nrm, "u_bump_strength": 1.0}
        fi_rail = {"u_col_tex": trim,     "u_nrm_tex": self._flat_nrm, "u_bump_strength": 0.0}
        posts = []
        n_posts = max(2, int(bw / 0.85))
        for pi in range(n_posts + 1):
            px = bx0 + pi * bw / n_posts
            posts.append(_box(px-.035, px+.035, -hd-bd-.04, -hd-bd+.04, bz, bz+.90, ts=0.4))
        posts.append(_box(bx0-.04, bx1+.04, -hd-bd-.04, -hd-bd+.04, bz+.85, bz+.96, ts=0.4))
        _attach(root, "bal_slab", _box(bx0, bx1, -hd-bd, -hd, bz-.18, bz, ts=1.0), sh, fi_wall)
        _attach(root, "bal_rail", _merge(*posts), sh, fi_rail)

    def _chimney(self, root, rng, hw, hd, h):
        sh = self._sh
        cw = rng.uniform(0.40, 0.65); cd = rng.uniform(0.40, 0.65)
        cx = rng.uniform(-hw*0.35, hw*0.35)
        top = h + self.PITCH*hd + rng.uniform(0.55, 1.20)
        cap_tex = self._s(55, 52, 50)
        fi_brick = {"u_col_tex": self._brick_col, "u_nrm_tex": self._brick_nrm, "u_bump_strength": 0.5}
        _attach(root, "chimney",
                _box(cx-cw*.5, cx+cw*.5, -cd*.5, cd*.5, h-.40, top, ts=0.7),
                sh, fi_brick)
        _attach(root, "chimney_cap",
                _box(cx-cw*.5-.07, cx+cw*.5+.07, -cd*.5-.07, cd*.5+.07, top-.10, top+.08, ts=0.4),
                sh, {"u_col_tex": cap_tex, "u_nrm_tex": self._flat_nrm, "u_bump_strength": 0.0})

    def _ac(self, root, rng, hw, hd, floors):
        sh = self._sh
        side = rng.choice(["left", "right"])
        az   = rng.randint(0, floors-1) * self.FLOOR_H + 0.80
        ay0  = rng.uniform(-hd+0.5, max(-hd+0.5, hd-1.1)); ay1 = ay0 + 0.60
        acc  = self._s(195, 195, 192)
        fi   = {"u_col_tex": acc, "u_nrm_tex": self._flat_nrm, "u_bump_strength": 0.0}
        if side == "right":
            geo = _box(hw-.04, hw+.52, ay0, ay1, az, az+.42, ts=0.4)
        else:
            geo = _box(-hw-.52, -hw+.04, ay0, ay1, az, az+.42, ts=0.4)
        _attach(root, "ac", geo, sh, fi)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _s(self, r: int, g: int, b: int):
        """Return a cached 1×1 solid-colour Texture — never allocates duplicates."""
        key = (r, g, b)
        if key not in self._solid_cache:
            from panda3d.core import Texture
            tex = Texture()
            tex.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb)
            tex.setRamImage(bytes([r, g, b]))
            self._solid_cache[key] = tex
        return self._solid_cache[key]

    def _wall_tex(self, rng: Random) -> tuple:
        """Pick one (col_tex, nrm_tex) from the wall material pool.

        Always the first RNG call in every build_* so all LOD levels share the
        same material for a given building seed.
        """
        return rng.choice(self._wall_mats)
