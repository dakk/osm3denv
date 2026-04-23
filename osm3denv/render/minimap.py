"""Dynamic circular OSM minimap overlay (GTA-style, bottom-right HUD).

A 3×3 tile grid is stitched into one texture.  Every frame:
  - the map is sampled with a rotated UV so the camera heading is always up
  - a small "N" indicator orbits the circle edge to mark where north is
  - the map is clipped to a circle by the fragment shader

Falls back to a single tile if PIL/Pillow is unavailable.
"""
from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path

import numpy as np
import requests
from direct.task import Task

log = logging.getLogger(__name__)

_TILE_URL   = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_USER_AGENT = "osm3denv/0.2 (+https://github.com/local/osm3denv)"
_EARTH_R    = 6_378_137.0


# ── tile maths ────────────────────────────────────────────────────────────────

def _tile_coords(lat: float, lon: float, z: int) -> tuple[int, int]:
    n  = 2 ** z
    tx = int((lon + 180.0) / 360.0 * n)
    ty = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return tx, ty


def _tile_frac(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Continuous tile position — integer part = tile index, fraction = offset within."""
    n = 2 ** z
    u = (lon + 180.0) / 360.0 * n
    v = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return u, v


def _tile_size_m(lat: float, z: int) -> float:
    return 2 * math.pi * _EARTH_R * math.cos(math.radians(lat)) / (2 ** z)


def _best_zoom(lat: float, radius_m: float) -> int:
    for z in range(16, 8, -1):
        if _tile_size_m(lat, z) >= radius_m * 2.5:
            return z
    return 9


# ── Minimap ───────────────────────────────────────────────────────────────────

class Minimap:
    """Downloads a 3×3 OSM tile grid and renders a circular heading-up overlay."""

    SIZE   = 0.30   # circle radius in aspect2d units
    MARGIN = 0.05   # gap from screen edge in aspect2d units

    def __init__(self, lat: float, lon: float, radius_m: float,
                 cache_dir: Path) -> None:
        self._lat       = lat
        self._lon       = lon
        self._radius_m  = radius_m
        self._cache_dir = cache_dir
        self._zoom      = _best_zoom(lat, radius_m)
        self._tile_path: Path | None = None
        self._tx0     = 0
        self._ty0     = 0
        self._n_tiles = 3   # 3 if stitched, 1 if PIL unavailable

    # ── public API ───────────────────────────────────────────────────────────

    def fetch(self) -> None:
        """Download (and cache) the 3×3 tile grid around the scene centre."""
        tx, ty = _tile_coords(self._lat, self._lon, self._zoom)
        self._tx0 = tx
        self._ty0 = ty

        stitched_path = self._cache_dir / f"minimap_s_{self._zoom}_{tx}_{ty}.png"
        if stitched_path.exists():
            log.debug("minimap stitched cache hit z=%d x=%d y=%d", self._zoom, tx, ty)
            self._tile_path = stitched_path
            return

        stitched_path.parent.mkdir(parents=True, exist_ok=True)

        tile_paths: dict[tuple[int, int], Path] = {}
        for dty in range(-1, 2):
            for dtx in range(-1, 2):
                ttx, tty = tx + dtx, ty + dty
                dest = self._cache_dir / f"minimap_{self._zoom}_{ttx}_{tty}.png"
                if not dest.exists():
                    url = _TILE_URL.format(z=self._zoom, x=ttx, y=tty)
                    log.info("downloading minimap tile z=%d x=%d y=%d",
                             self._zoom, ttx, tty)
                    try:
                        r = requests.get(url, headers={"User-Agent": _USER_AGENT},
                                         timeout=15)
                        r.raise_for_status()
                        dest.write_bytes(r.content)
                    except requests.RequestException as exc:
                        log.warning("minimap tile %d/%d/%d failed: %s",
                                    self._zoom, ttx, tty, exc)
                        continue
                tile_paths[(dtx, dty)] = dest

        self._stitch(tile_paths, stitched_path)

    def attach_to(self, base) -> None:
        """Attach the circular minimap to *base.aspect2d* and register the update task."""
        if self._tile_path is None:
            log.warning("minimap: tile unavailable, skipping overlay")
            return

        from panda3d.core import CardMaker, Filename, LineSegs, Texture

        from osm3denv.render.helpers import load_shader

        tex = Texture("minimap")
        tex.read(Filename.fromOsSpecific(str(self._tile_path)))
        tex.setWrapU(Texture.WM_border_color)
        tex.setWrapV(Texture.WM_border_color)
        tex.setBorderColor((0.06, 0.06, 0.08, 1.0))

        # Card sits in the bottom-right; SIZE is both radius and half-side.
        ar = base.getAspectRatio()
        s, mg = self.SIZE, self.MARGIN
        xl = ar - mg - s * 2;  xr = ar - mg
        zb = -1.0 + mg;        zt = -1.0 + mg + s * 2
        cx = (xl + xr) * 0.5;  cz = (zb + zt) * 0.5

        cm = CardMaker("minimap_card")
        cm.setFrame(xl, xr, zb, zt)
        card = base.aspect2d.attachNewNode(cm.generate())
        card.setBin("fixed", 10)

        shader = load_shader("minimap")
        if shader:
            card.setShader(shader)
            card.setShaderInput("u_map_tex",   tex)
            card.setShaderInput("u_cam_uv",    (0.5, 0.5))
            card.setShaderInput("u_heading",   0.0)
            card.setShaderInput("u_map_scale", 1.0 / self._n_tiles)
        else:
            card.setTexture(tex)

        # ── player chevron — fixed at centre, always pointing up (= forward) ──
        player = self._make_player_chevron(base.aspect2d, s * 0.095)
        player.setPos(cx, 0, cz)
        player.setBin("fixed", 14)

        # ── "N" north indicator — small red triangle at circle edge ──────────
        # Geometry is a small down-pointing triangle centred at origin;
        # we reposition it every frame so it orbits the circle to mark north.
        north_np = self._make_north_mark(base.aspect2d, s * 0.055)
        north_np.setBin("fixed", 13)

        self._base     = base
        self._card     = card
        self._cx       = cx
        self._cz       = cz
        self._s        = s
        self._north_np = north_np
        self._shader   = shader

        base.taskMgr.add(self._update_task, "minimap_update")
        log.info("minimap attached (zoom=%d, %d×%d tiles)",
                 self._zoom, self._n_tiles, self._n_tiles)

    # ── internal ─────────────────────────────────────────────────────────────

    def _stitch(self, tile_paths: dict, stitched_path: Path) -> None:
        try:
            from PIL import Image
            canvas = Image.new("RGB", (768, 768), (30, 30, 30))
            for (dtx, dty), path in tile_paths.items():
                try:
                    tile = Image.open(str(path)).convert("RGB").resize((256, 256))
                    canvas.paste(tile, ((dtx + 1) * 256, (dty + 1) * 256))
                except Exception as exc:
                    log.warning("minimap stitch tile (%d,%d): %s", dtx, dty, exc)
            canvas.save(str(stitched_path))
            self._tile_path = stitched_path
            self._n_tiles   = 3
            log.info("minimap stitched (zoom=%d)", self._zoom)
        except ImportError:
            centre = tile_paths.get((0, 0))
            if centre:
                shutil.copy(str(centre), str(stitched_path))
                self._tile_path = stitched_path
                self._n_tiles   = 1
                log.warning("PIL unavailable; minimap uses single tile")

    def _make_player_chevron(self, parent, size: float):
        """White upward chevron at origin — player position / forward indicator."""
        from panda3d.core import LineSegs
        segs = LineSegs("minimap_player")
        segs.setColor(1.0, 1.0, 1.0, 1.0)
        segs.setThickness(2.0)
        # V-shape pointing up
        segs.moveTo(-size,       0, -size * 0.55)
        segs.drawTo( 0,          0,  size)
        segs.drawTo( size,       0, -size * 0.55)
        np_ = parent.attachNewNode(segs.create())
        return np_

    def _make_north_mark(self, parent, size: float):
        """Small red downward triangle centred at origin — repositioned each frame."""
        from panda3d.core import LineSegs
        segs = LineSegs("minimap_north")
        segs.setColor(0.95, 0.25, 0.25, 1.0)
        segs.setThickness(2.0)
        segs.moveTo( 0,     0,  size)
        segs.drawTo(-size,  0, -size * 0.5)
        segs.drawTo( size,  0, -size * 0.5)
        segs.drawTo( 0,     0,  size)
        np_ = parent.attachNewNode(segs.create())
        return np_

    def _update_task(self, task) -> int:
        base = self._base
        if getattr(base, "_frame", None) is None:
            return Task.cont

        # ── camera world position → lat/lon ──────────────────────────────────
        pos = base.camera.getPos()
        lon_arr, lat_arr = base._frame.to_ll(
            np.array([pos.x], dtype=np.float64),
            np.array([pos.y], dtype=np.float64),
        )
        lat = float(lat_arr[0])
        lon = float(lon_arr[0])

        # ── camera position as UV in stitched texture ─────────────────────────
        # nt tiles wide; left edge starts at tx0 - nt//2
        u_cam, v_cam = _tile_frac(lat, lon, self._zoom)
        nt   = self._n_tiles
        half = nt // 2
        u_frac = (u_cam - (self._tx0 - half)) / nt
        v_frac = (v_cam - (self._ty0 - half)) / nt
        v_gl   = 1.0 - v_frac          # flip: V=0=south, V=1=north (OpenGL)

        # ── update shader uniforms ────────────────────────────────────────────
        if self._shader:
            hrad = math.radians(base.heading)
            self._card.setShaderInput("u_cam_uv",  (u_frac, v_gl))
            self._card.setShaderInput("u_heading", hrad)

            # ── north indicator position on the circle edge ───────────────────
            # Map rotates CCW by heading; north (originally at top) moves to
            # angle (90° + heading°) measured CCW from screen +X axis.
            # In aspect2d (X=right, Z=up): cos(90+h)=-sin(h), sin(90+h)=cos(h).
            r_edge = self._s * 0.82       # slightly inside the rim
            nx = self._cx - r_edge * math.sin(hrad)
            nz = self._cz + r_edge * math.cos(hrad)
            self._north_np.setPos(nx, 0, nz)

        return Task.cont
