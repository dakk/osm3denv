"""Panda3D viewer.

Controls
--------
* **W / A / S / D** — move forward / left / back / right.
* **Q / E** — move down / up.
* **Shift** — sprint (x4 speed).
* **Right-drag** — look around.
* **Mouse wheel** — cycle move speed.
* **T** — pause / resume time.
* **[ / ]** — jump time ±1 hour.
* **Escape** — quit.
"""
from __future__ import annotations

import logging
import math
import sys

import numpy as np
from direct.filter.CommonFilters import CommonFilters
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight, DirectionalLight,
    LVector3, LVector3f, TextNode, Vec4, WindowProperties,
    loadPrcFileData,
)

from osm3denv.entity import MapEntity
from osm3denv.entities.terrain import Terrain

log = logging.getLogger(__name__)

# ── day/night helpers ─────────────────────────────────────────────────────────

def _c01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

def _sun_params(tod: float):
    """Return (sun_dir, sun_color, amb_color, sky_color) for time_of_day ∈ [0,1].

    tod=0/1 = midnight, tod=0.25 = sunrise, tod=0.5 = noon, tod=0.75 = sunset.
    """
    phase  = 2.0 * math.pi * tod
    sin_el = -math.cos(phase)          # elevation sine: -1=midnight, +1=noon
    cos_el = math.sqrt(max(0.0, 1.0 - sin_el ** 2))

    # Sun sweeps from east (sunrise) through south to west (sunset).
    az_angle = math.pi * (tod - 0.25) / 0.5   # 0 at sunrise, π at sunset
    sun_dir = LVector3f(
        math.sin(az_angle) * cos_el,   # east component
        math.cos(az_angle) * cos_el,   # north component
        sin_el,                         # up component
    )

    # Sun color ----------------------------------------------------------------
    t_rise = _c01((sin_el + 0.05) / 0.12)   # 0 below horizon → 1 just above
    t_high = _c01((sin_el - 0.15) / 0.20)   # 0 near horizon → 1 in full day
    sun_c  = _lerp((0.0, 0.0, 0.0), (1.0, 0.50, 0.08), t_rise)
    sun_c  = _lerp(sun_c,            (0.95, 0.92, 0.85), t_high)

    # Ambient color ------------------------------------------------------------
    t_a1  = _c01((sin_el + 0.20) / 0.35)
    t_a2  = _c01((sin_el - 0.05) / 0.20)
    amb_c = _lerp((0.03, 0.03, 0.08), (0.22, 0.18, 0.15), t_a1)
    amb_c = _lerp(amb_c,              (0.35, 0.37, 0.42), t_a2)

    # Sky color ----------------------------------------------------------------
    t_s1  = _c01((sin_el + 0.15) / 0.20)
    t_s2  = _c01((sin_el - 0.05) / 0.20)
    sky_c = _lerp((0.01, 0.01, 0.05), (0.75, 0.30, 0.05), t_s1)
    sky_c = _lerp(sky_c,              (0.53, 0.70, 0.86), t_s2)

    return sun_dir, sun_c, amb_c, sky_c


# ── viewer ────────────────────────────────────────────────────────────────────

class TerrainViewer(ShowBase):
    MOVE_KEYS  = ("w", "a", "s", "d", "q", "e")
    TIME_SPEED = 1.0 / 300.0   # 1 real second = 1/300 of a day ≈ 4.8 min/s

    def __init__(self, terrain: Terrain,
                 entities: list[MapEntity] | None = None,
                 frame=None, minimap=None) -> None:
        loadPrcFileData("", "framebuffer-multisample 1\nmultisamples 4")
        ShowBase.__init__(self)
        self._frame = frame
        self._origin_alt_m = float(terrain.data.origin_alt_m)

        props = WindowProperties()
        props.setTitle("osm3denv — terrain")
        self.win.requestProperties(props)

        for entity in (entities or []):
            entity.attach_to(self.render)

        # Panda3D lights (affect fixed-function / auto-shader geometry)
        self._ambient_light = AmbientLight("ambient")
        self.render.setLight(self.render.attachNewNode(self._ambient_light))
        self._sun_light   = DirectionalLight("sun")
        self._sun_np      = self.render.attachNewNode(self._sun_light)
        self.render.setLight(self._sun_np)

        filters = CommonFilters(self.win, self.cam)
        if filters.setBloom(size="large", mintrigger=0.4, maxtrigger=0.8,
                            intensity=0.6, desat=0.3):
            self._filters = filters
        else:
            log.warning("bloom post-processing not supported on this GPU")
            self._filters = None

        r = float(terrain.data.radius_m)
        self.disableMouse()
        self.camera.setPos(0.0, -r * 1.5, r * 0.8)
        self.heading = 0.0
        self.pitch   = -25.0
        self.camera.setHpr(self.heading, self.pitch, 0)
        self.camLens.setFar(max(20_000.0, r * 20.0))
        self.camLens.setNear(1.0)
        self.camLens.setFov(70)

        # Time of day: 0=midnight, 0.25=sunrise, 0.5=noon, 0.75=sunset
        self.time_of_day   = 0.5    # start at noon
        self.time_running  = True
        self._apply_time_of_day()   # set initial shader inputs + background

        self.move_speed = max(50.0, r * 0.2)
        self.shift_held = False
        self.keys: dict[str, bool] = {k: False for k in self.MOVE_KEYS}
        for k in self.MOVE_KEYS:
            self.accept(k, self._set_key, [k, True])
            self.accept(k + "-up", self._set_key, [k, False])
        self.accept("shift",    self._set_shift, [True])
        self.accept("shift-up", self._set_shift, [False])
        self.accept("wheel_up",   self._bump_speed, [1.25])
        self.accept("wheel_down", self._bump_speed, [0.8])
        self.accept("escape", sys.exit)
        self.accept("t",   self._toggle_time)
        self.accept("[",   self._jump_time, [-1.0 / 24.0])
        self.accept("]",   self._jump_time, [ 1.0 / 24.0])

        self.looking = False
        self._last_mouse: tuple[float, float] | None = None
        self.accept("mouse3",    self._start_look)
        self.accept("mouse3-up", self._stop_look)

        OnscreenText(
            text=("WASD/QE move  ·  right-drag look  ·  Shift sprint  ·  "
                  "wheel speed  ·  T pause time  ·  [ ] ±1h  ·  Esc quit"),
            pos=(-1.3, -0.95), scale=0.04,
            fg=(1, 1, 1, 0.9), bg=(0, 0, 0, 0.4),
            align=TextNode.ALeft, mayChange=False,
        )
        self.speed_text = OnscreenText(
            text="", pos=(1.3, -0.95), scale=0.04,
            fg=(1, 1, 1, 0.9), bg=(0, 0, 0, 0.4),
            align=TextNode.ARight, mayChange=True,
        )
        self._refresh_speed_text()

        self.pos_text = OnscreenText(
            text="", pos=(-1.3, 0.95), scale=0.04,
            fg=(1, 1, 1, 0.9), bg=(0, 0, 0, 0.4),
            align=TextNode.ALeft, mayChange=True,
        )

        self.time_text = OnscreenText(
            text="", pos=(1.3, 0.95), scale=0.05,
            fg=(1, 1, 1, 0.95), bg=(0, 0, 0, 0.4),
            align=TextNode.ARight, mayChange=True,
        )
        self._refresh_time_text()

        self.fps_text = OnscreenText(
            text="", pos=(0.0, 0.95), scale=0.045,
            fg=(1, 1, 1, 0.9), bg=(0, 0, 0, 0.4),
            align=TextNode.ACenter, mayChange=True,
        )

        if minimap is not None:
            minimap.attach_to(self)

        self.taskMgr.add(self._update, "camera_update")

    # ── time controls ─────────────────────────────────────────────────────────

    def _toggle_time(self) -> None:
        self.time_running = not self.time_running

    def _jump_time(self, delta: float) -> None:
        self.time_of_day = (self.time_of_day + delta) % 1.0
        self._apply_time_of_day()
        self._refresh_time_text()

    def _apply_time_of_day(self) -> None:
        sun_dir, sun_c, amb_c, sky_c = _sun_params(self.time_of_day)

        # Global shader inputs — propagate to all custom shaders on render
        self.render.setShaderInput("u_sun_dir",   sun_dir)
        self.render.setShaderInput("u_sun_color", LVector3f(*sun_c))
        self.render.setShaderInput("u_amb_color", LVector3f(*amb_c))
        self.render.setShaderInput("u_sky_color", LVector3f(*sky_c))

        # Panda3D lights (for any non-custom-shader geometry)
        self._sun_light.setColor(Vec4(*sun_c, 1.0))
        self._ambient_light.setColor(Vec4(*amb_c, 1.0))
        el = math.degrees(math.asin(max(-1.0, min(1.0, sun_dir.z))))
        self._sun_np.setHpr(math.degrees(math.atan2(sun_dir.x, sun_dir.y)), -el, 0)

        # setBackgroundColor only affects the main window clear colour.
        # After CommonFilters, the scene renders into an offscreen buffer whose
        # clear colour is what actually shows as "sky" — update that too.
        self.setBackgroundColor(*sky_c, 1.0)
        if self._filters is not None:
            from panda3d.core import LVector4f
            clear = LVector4f(*sky_c, 1.0)
            for buf in self._filters.manager.buffers:
                buf.setClearColor(clear)

    def _refresh_time_text(self) -> None:
        mins   = int(self.time_of_day * 24 * 60) % (24 * 60)
        hh, mm = divmod(mins, 60)
        self.time_text.setText(f"{hh:02d}:{mm:02d}")

    # ── misc controls ─────────────────────────────────────────────────────────

    def _set_key(self, k: str, v: bool) -> None:   self.keys[k] = v
    def _set_shift(self, v: bool) -> None:          self.shift_held = v

    def _bump_speed(self, factor: float) -> None:
        self.move_speed = max(1.0, min(10_000.0, self.move_speed * factor))
        self._refresh_speed_text()

    def _refresh_speed_text(self) -> None:
        self.speed_text.setText(f"speed: {self.move_speed:.0f} m/s")

    def _start_look(self) -> None:
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            self._last_mouse = (m.getX(), m.getY())
            self.looking = True

    def _stop_look(self) -> None:
        self.looking = False
        self._last_mouse = None

    # ── main update ───────────────────────────────────────────────────────────

    def _update(self, task: Task.Task) -> int:
        dt = globalClock.getDt()  # noqa: F821

        # Advance time of day
        if self.time_running:
            self.time_of_day = (self.time_of_day + self.TIME_SPEED * dt) % 1.0
            self._apply_time_of_day()
            self._refresh_time_text()

        if self.looking and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            if self._last_mouse is not None:
                dx = m.getX() - self._last_mouse[0]
                dy = m.getY() - self._last_mouse[1]
                self.heading -= dx * 180.0
                self.pitch = max(-89.0, min(89.0, self.pitch + dy * 180.0))
                self.camera.setHpr(self.heading, self.pitch, 0)
            self._last_mouse = (m.getX(), m.getY())

        h       = math.radians(self.heading)
        forward = LVector3(-math.sin(h), math.cos(h), 0.0)
        right   = LVector3( math.cos(h), math.sin(h), 0.0)
        up      = LVector3(0.0, 0.0, 1.0)

        move = LVector3(0.0, 0.0, 0.0)
        if self.keys["w"]: move += forward
        if self.keys["s"]: move -= forward
        if self.keys["d"]: move += right
        if self.keys["a"]: move -= right
        if self.keys["e"]: move += up
        if self.keys["q"]: move -= up

        if move.lengthSquared() > 0.0:
            move.normalize()
            speed = self.move_speed * (4.0 if self.shift_held else 1.0)
            self.camera.setPos(self.camera.getPos() + move * speed * dt)

        if self._frame is not None:
            pos = self.camera.getPos()
            lon_arr, lat_arr = self._frame.to_ll(
                np.array([pos.x], dtype=np.float64),
                np.array([pos.y], dtype=np.float64),
            )
            alt = pos.z + self._origin_alt_m
            self.pos_text.setText(
                f"lat {lat_arr[0]:.6f}°  lon {lon_arr[0]:.6f}°  alt {alt:.0f} m"
            )

        self.fps_text.setText(f"{globalClock.getAverageFrameRate():.0f} fps")  # noqa: F821

        return Task.cont


def run_viewer(terrain: Terrain,
               entities: list[MapEntity] | None = None,
               frame=None, minimap=None) -> None:
    TerrainViewer(terrain, entities=entities, frame=frame, minimap=minimap).run()
