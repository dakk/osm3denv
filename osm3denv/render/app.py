"""Minimal Panda3D viewer for a terrain scene.

Controls
--------
* **W / A / S / D** — move forward / left / back / right (along camera heading, ground plane).
* **Q / E** — move down / up (world Z).
* **Shift** — sprint (x4 speed).
* **Hold right mouse button** — drag to look around.
* **Mouse wheel** — cycle move speed.
* **Escape** — quit.
"""
from __future__ import annotations

import logging
import math
import sys

import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight,
    DepthOffsetAttrib,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    LVector3,
    TextNode,
    Vec4,
    WindowProperties,
)
from direct.gui.OnscreenText import OnscreenText

from osm3denv.layer import RenderLayer
from osm3denv.mesh.terrain import TerrainData

log = logging.getLogger(__name__)


def _layer_to_np(layer: RenderLayer, parent) -> None:
    """Attach a RenderLayer to *parent* as one or two Panda3D nodes."""
    if layer.vertices is not None and layer.normals is not None:
        has_uvs = layer.uvs is not None
        vfmt = GeomVertexFormat.getV3n3t2() if has_uvs else GeomVertexFormat.getV3n3()
        vdata = GeomVertexData(layer.name, vfmt, Geom.UHStatic)
        n = len(layer.vertices)
        vdata.setNumRows(n)

        vwriter = GeomVertexWriter(vdata, "vertex")
        nwriter = GeomVertexWriter(vdata, "normal")
        if has_uvs:
            twriter = GeomVertexWriter(vdata, "texcoord")

        verts = layer.vertices
        norms = layer.normals
        for i in range(n):
            vwriter.addData3(float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2]))
            nwriter.addData3(float(norms[i, 0]), float(norms[i, 1]), float(norms[i, 2]))
            if has_uvs:
                twriter.addData2(float(layer.uvs[i, 0]), float(layer.uvs[i, 1]))

        prim = GeomTriangles(Geom.UHStatic)
        if layer.indices is not None:
            idx = np.asarray(layer.indices, dtype=np.uint32).reshape(-1, 3)
            for a, b, c in idx:
                prim.addVertices(int(a), int(b), int(c))
        else:
            for i in range(0, n, 3):
                prim.addVertices(i, i + 1, i + 2)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        geom_node = GeomNode(layer.name)
        geom_node.addGeom(geom)
        if layer.depth_offset:
            geom_node.setAttrib(DepthOffsetAttrib.make(layer.depth_offset))

        mesh_np = parent.attachNewNode(geom_node)
        mesh_np.setColor(Vec4(*layer.color))
        if not layer.lit:
            mesh_np.setLightOff()
        if layer.two_sided:
            mesh_np.setTwoSided(True)

    if layer.polylines is not None:
        ls = LineSegs(layer.name + "_lines")
        ls.setColor(*layer.color)
        ls.setThickness(layer.line_thickness)
        for polyline in layer.polylines:
            if len(polyline) < 2:
                continue
            ls.moveTo(float(polyline[0, 0]), float(polyline[0, 1]), float(polyline[0, 2]))
            for pt in polyline[1:]:
                ls.drawTo(float(pt[0]), float(pt[1]), float(pt[2]))
        line_np = parent.attachNewNode(ls.create())
        if not layer.lit:
            line_np.setLightOff()


class TerrainViewer(ShowBase):
    MOVE_KEYS = ("w", "a", "s", "d", "q", "e")

    def __init__(self, terrain: TerrainData,
                 layers: list[RenderLayer] | None = None):
        ShowBase.__init__(self)

        props = WindowProperties()
        props.setTitle("osm3denv — terrain")
        self.win.requestProperties(props)

        self.setBackgroundColor(0.53, 0.70, 0.86, 1.0)

        for layer in (layers or []):
            _layer_to_np(layer, self.render)

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.35, 0.35, 0.40, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.95, 0.92, 0.85, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-30, -50, 0)
        self.render.setLight(sun_np)

        r = float(terrain.radius_m)
        self.disableMouse()
        self.camera.setPos(0.0, -r * 1.5, r * 0.8)
        self.heading = 0.0
        self.pitch = -25.0
        self.camera.setHpr(self.heading, self.pitch, 0)
        self.camLens.setFar(max(20_000.0, r * 20.0))
        self.camLens.setNear(1.0)
        self.camLens.setFov(70)

        self.move_speed = max(50.0, r * 0.2)
        self.shift_held = False
        self.keys: dict[str, bool] = {k: False for k in self.MOVE_KEYS}
        for k in self.MOVE_KEYS:
            self.accept(k, self._set_key, [k, True])
            self.accept(k + "-up", self._set_key, [k, False])
        self.accept("shift", self._set_shift, [True])
        self.accept("shift-up", self._set_shift, [False])
        self.accept("wheel_up", self._bump_speed, [1.25])
        self.accept("wheel_down", self._bump_speed, [0.8])
        self.accept("escape", sys.exit)

        self.looking = False
        self._last_mouse: tuple[float, float] | None = None
        self.accept("mouse3", self._start_look)
        self.accept("mouse3-up", self._stop_look)

        OnscreenText(
            text="WASD/QE move  ·  right-drag look  ·  Shift sprint  ·  wheel speed  ·  Esc quit",
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

        self.taskMgr.add(self._update, "camera_update")

    def _set_key(self, k: str, v: bool) -> None:
        self.keys[k] = v

    def _set_shift(self, v: bool) -> None:
        self.shift_held = v

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

    def _update(self, task: Task.Task) -> int:
        dt = globalClock.getDt()  # noqa: F821 — panda3d injects globalClock

        if self.looking and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            if self._last_mouse is not None:
                dx = m.getX() - self._last_mouse[0]
                dy = m.getY() - self._last_mouse[1]
                self.heading -= dx * 180.0
                self.pitch = max(-89.0, min(89.0, self.pitch + dy * 180.0))
                self.camera.setHpr(self.heading, self.pitch, 0)
            self._last_mouse = (m.getX(), m.getY())

        h = math.radians(self.heading)
        forward = LVector3(-math.sin(h), math.cos(h), 0.0)
        right = LVector3(math.cos(h), math.sin(h), 0.0)
        up = LVector3(0.0, 0.0, 1.0)

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

        return Task.cont


def run_viewer(terrain: TerrainData,
               layers: list[RenderLayer] | None = None) -> None:
    TerrainViewer(terrain, layers=layers).run()
