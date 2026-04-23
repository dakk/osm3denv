"""Dynamic circular OSM minimap overlay (GTA-style, bottom-right HUD).

Tile zoom levels
----------------
Four zoom levels are pre-fetched:  z-1, z, z+1, z+2  where z is the base
zoom chosen for the scene radius.  The active level is selected each frame
from camera altitude so that zooming in switches to more-detailed tiles and
zooming out switches to coarser tiles.  Each level uses a 3×3 stitched grid
so the camera can roam within the scene without leaving the covered area.
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
    """Continuous tile position — integer part = index, fraction = intra-tile offset."""
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
    """Downloads 3×3 OSM tile grids at several zoom levels and renders a
    circular heading-up overlay whose tile zoom tracks camera altitude."""

    SIZE   = 0.30
    MARGIN = 0.05

    def __init__(self, lat: float, lon: float, radius_m: float,
                 cache_dir: Path) -> None:
        self._lat       = lat
        self._lon       = lon
        self._radius_m  = radius_m
        self._cache_dir = cache_dir
        self._zoom      = _best_zoom(lat, radius_m)
        self._n_tiles   = 3   # 3 if PIL available, 1 otherwise

        # Populated by fetch(); keyed by OSM zoom level.
        self._stitched_by_zoom: dict[int, Path] = {}
        self._centers: dict[int, tuple[int, int]] = {}   # {z: (tx, ty)}

        # Kept for backward-compat (attach_to checks this before the dict).
        self._tile_path: Path | None = None

    # ── public API ───────────────────────────────────────────────────────────

    def fetch(self) -> None:
        """Download 3×3 tile grids for z-1, z, z+1, z+2 and stitch each."""
        z0 = self._zoom
        zoom_levels = sorted({max(1, z0 - 1), z0,
                               min(19, z0 + 1), min(19, z0 + 2)})

        for z in zoom_levels:
            tx, ty = _tile_coords(self._lat, self._lon, z)
            self._centers[z] = (tx, ty)

            stitched = self._cache_dir / f"minimap_s_{z}_{tx}_{ty}.png"
            if stitched.exists():
                log.debug("minimap cache hit z=%d x=%d y=%d", z, tx, ty)
                self._stitched_by_zoom[z] = stitched
                continue

            stitched.parent.mkdir(parents=True, exist_ok=True)
            tile_paths: dict[tuple[int, int], Path] = {}

            for dty in range(-1, 2):
                for dtx in range(-1, 2):
                    dest = self._cache_dir / f"minimap_{z}_{tx+dtx}_{ty+dty}.png"
                    if not dest.exists():
                        url = _TILE_URL.format(z=z, x=tx + dtx, y=ty + dty)
                        log.info("downloading minimap tile z=%d x=%d y=%d",
                                 z, tx + dtx, ty + dty)
                        try:
                            r = requests.get(
                                url, headers={"User-Agent": _USER_AGENT}, timeout=15)
                            r.raise_for_status()
                            dest.write_bytes(r.content)
                        except requests.RequestException as exc:
                            log.warning("minimap tile %d/%d/%d failed: %s",
                                        z, tx + dtx, ty + dty, exc)
                            continue
                    tile_paths[(dtx, dty)] = dest

            self._stitch(tile_paths, stitched)
            if stitched.exists():
                self._stitched_by_zoom[z] = stitched

        self._tile_path = self._stitched_by_zoom.get(z0)
        log.info("minimap fetched zoom levels %s", sorted(self._stitched_by_zoom))

    def attach_to(self, base) -> None:
        """Attach the circular minimap to *base.aspect2d* and register the update task."""
        if not self._stitched_by_zoom:
            log.warning("minimap: no tiles available, skipping overlay")
            return

        from panda3d.core import CardMaker, Filename, Texture

        from osm3denv.render.helpers import load_shader

        # Pre-load one Panda3D Texture per zoom level.
        self._textures: dict[int, object] = {}
        for z, path in self._stitched_by_zoom.items():
            tex = Texture(f"minimap_{z}")
            tex.read(Filename.fromOsSpecific(str(path)))
            tex.setWrapU(Texture.WM_border_color)
            tex.setWrapV(Texture.WM_border_color)
            tex.setBorderColor((0.06, 0.06, 0.08, 1.0))
            self._textures[z] = tex

        z0  = self._zoom
        if z0 not in self._textures:
            z0 = min(self._textures)

        # Card — fixed position, bottom-right corner.
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
            card.setShaderInput("u_map_tex",   self._textures[z0])
            card.setShaderInput("u_cam_uv",    (0.5, 0.5))
            card.setShaderInput("u_heading",   0.0)
            card.setShaderInput("u_map_scale", 1.0 / self._n_tiles)
        else:
            card.setTexture(self._textures[z0])

        # Player chevron — fixed at centre, always pointing up (= forward).
        player = self._make_player_chevron(base.aspect2d, s * 0.095)
        player.setPos(cx, 0, cz)
        player.setBin("fixed", 14)

        # North indicator — small triangle repositioned each frame.
        north_np = self._make_north_mark(base.aspect2d, s * 0.055)
        north_np.setBin("fixed", 13)

        self._base         = base
        self._card         = card
        self._cx           = cx
        self._cz           = cz
        self._s            = s
        self._north_np     = north_np
        self._shader       = shader
        self._active_zoom  = z0

        base.taskMgr.add(self._update_task, "minimap_update")
        log.info("minimap attached — zoom levels %s, n_tiles=%d",
                 sorted(self._textures), self._n_tiles)

    # ── internal ─────────────────────────────────────────────────────────────

    def _zoom_for_altitude(self, alt_m: float) -> int:
        """Pick the OSM zoom level that best matches the camera altitude.

        alt_ref is the altitude at which the base zoom level shows 1 tile.
        Each step up/down in zoom level doubles/halves the detail.
        """
        z0      = self._zoom
        alt_ref = max(150.0, self._radius_m * 0.20)

        if   alt_m < alt_ref * 0.12:  candidate = min(19, z0 + 2)
        elif alt_m < alt_ref * 0.50:  candidate = min(19, z0 + 1)
        elif alt_m > alt_ref * 2.00:  candidate = max(1,  z0 - 1)
        else:                          candidate = z0

        # Fall back to nearest available level if the candidate wasn't fetched.
        if candidate in self._textures:
            return candidate
        available = sorted(self._textures)
        return min(available, key=lambda z: abs(z - candidate))

    def _stitch(self, tile_paths: dict, stitched_path: Path) -> None:
        try:
            from PIL import Image
            canvas = Image.new("RGB", (768, 768), (30, 30, 30))
            for (dtx, dty), path in tile_paths.items():
                try:
                    tile = Image.open(str(path)).convert("RGB").resize((256, 256))
                    canvas.paste(tile, ((dtx + 1) * 256, (dty + 1) * 256))
                except Exception as exc:
                    log.warning("minimap stitch (%d,%d): %s", dtx, dty, exc)
            canvas.save(str(stitched_path))
            self._n_tiles = 3
        except ImportError:
            centre = tile_paths.get((0, 0))
            if centre:
                shutil.copy(str(centre), str(stitched_path))
                self._n_tiles = 1
                log.warning("PIL unavailable; minimap uses single tiles")

    def _make_player_chevron(self, parent, size: float):
        from panda3d.core import LineSegs
        segs = LineSegs("minimap_player")
        segs.setColor(1.0, 1.0, 1.0, 1.0)
        segs.setThickness(2.0)
        segs.moveTo(-size, 0, -size * 0.55)
        segs.drawTo( 0,    0,  size)
        segs.drawTo( size, 0, -size * 0.55)
        return parent.attachNewNode(segs.create())

    def _make_north_mark(self, parent, size: float):
        from panda3d.core import LineSegs
        segs = LineSegs("minimap_north")
        segs.setColor(0.95, 0.25, 0.25, 1.0)
        segs.setThickness(2.0)
        segs.moveTo( 0,    0,  size)
        segs.drawTo(-size, 0, -size * 0.5)
        segs.drawTo( size, 0, -size * 0.5)
        segs.drawTo( 0,    0,  size)
        return parent.attachNewNode(segs.create())

    def _update_task(self, _) -> int:
        base = self._base
        if getattr(base, "_frame", None) is None:
            return Task.cont

        # Camera world position → lat/lon.
        pos = base.camera.getPos()
        lon_arr, lat_arr = base._frame.to_ll(
            np.array([pos.x], dtype=np.float64),
            np.array([pos.y], dtype=np.float64),
        )
        lat = float(lat_arr[0])
        lon = float(lon_arr[0])

        # Select zoom level from altitude; swap texture when it changes.
        target_z = self._zoom_for_altitude(float(pos.z))
        if target_z != self._active_zoom:
            self._card.setShaderInput("u_map_tex", self._textures[target_z])
            self._active_zoom = target_z

        # Camera UV in the stitched texture for the active zoom level.
        tx0, ty0 = self._centers[self._active_zoom]
        u_cam, v_cam = _tile_frac(lat, lon, self._active_zoom)
        nt   = self._n_tiles
        half = nt // 2
        u_frac = (u_cam - (tx0 - half)) / nt
        v_frac = (v_cam - (ty0 - half)) / nt
        v_gl   = 1.0 - v_frac

        hrad = math.radians(base.heading)

        # Always show 1 tile at the active zoom level; u_map_scale = 1/nt.
        self._card.setShaderInput("u_cam_uv",    (u_frac, v_gl))
        self._card.setShaderInput("u_heading",   hrad)
        self._card.setShaderInput("u_map_scale", 1.0 / nt)

        # North indicator orbits the circle edge opposite to map rotation.
        r_edge = self._s * 0.82
        self._north_np.setPos(
            self._cx - r_edge * math.sin(hrad),
            0,
            self._cz + r_edge * math.cos(hrad),
        )

        return Task.cont
