"""Minimal Panda3D viewer for a TerrainData mesh.

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

from osm3denv.mesh.coastline import CoastlineData
from osm3denv.mesh.terrain import TerrainData
from osm3denv.mesh.water import WaterData

log = logging.getLogger(__name__)


def _build_terrain_node(terrain: TerrainData) -> GeomNode:
    vfmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData("terrain", vfmt, Geom.UHStatic)
    n = len(terrain.vertices)
    vdata.setNumRows(n)

    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")
    twriter = GeomVertexWriter(vdata, "texcoord")

    verts = terrain.vertices
    norms = terrain.normals
    uvs = terrain.uvs
    for i in range(n):
        vx, vy, vz = verts[i]
        nx, ny, nz = norms[i]
        u, v = uvs[i]
        vwriter.addData3(float(vx), float(vy), float(vz))
        nwriter.addData3(float(nx), float(ny), float(nz))
        twriter.addData2(float(u), float(v))

    prim = GeomTriangles(Geom.UHStatic)
    idx = np.asarray(terrain.indices, dtype=np.uint32).reshape(-1, 3)
    for a, b, c in idx:
        prim.addVertices(int(a), int(b), int(c))
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("terrain")
    node.addGeom(geom)
    return node


def _triangulate_flat_poly(poly, max_seg: float) -> list[list[tuple[float, float]]]:
    """Densify *poly*, Delaunay-triangulate it, return CCW (x,y) triples.

    Two robustness fixes vs plain delaunay_triangles():
    * segmentize() prevents triangles that jump across concave pockets.
    * covers() (not contains()) keeps triangles whose centroid sits exactly
      on a boundary edge.
    """
    import shapely
    densified = poly.segmentize(max_seg)
    result = []
    for tri in shapely.delaunay_triangles(densified).geoms:
        if not poly.covers(tri.centroid):
            continue
        coords = list(tri.exterior.coords)[:-1]
        if len(coords) != 3:
            continue
        (x0, y0), (x1, y1), (x2, y2) = coords
        if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) < 0:
            coords = [coords[0], coords[2], coords[1]]
        result.append(coords)
    return result


def _sea_tris_inner(sea_polygon, radius_m: float) -> list[list[tuple[float, float]]]:
    """Triangulate the sea polygon clipped to the terrain bbox."""
    import shapely.geometry as sg
    inner_sea = sea_polygon.intersection(sg.box(-radius_m, -radius_m, radius_m, radius_m))
    if inner_sea.is_empty:
        return []
    return _triangulate_flat_poly(inner_sea, max_seg=max(30.0, radius_m / 30.0))


def _sea_tris_outer(radius_m: float, extent: float) -> list[list[tuple[float, float]]]:
    """8 CCW triangles covering the rectangular frame between terrain bbox and extent.

    Avoids Shapely entirely for this part — the frame is trivially decomposed
    into 4 trapezoid quads (2 triangles each), all pre-verified CCW.
    """
    r, e = radius_m, extent
    return [
        # North strip
        [(-e, e), (r, r), (e, e)],
        [(-e, e), (-r, r), (r, r)],
        # South strip
        [(-e, -e), (e, -e), (r, -r)],
        [(-e, -e), (r, -r), (-r, -r)],
        # West strip
        [(-e, -e), (-r, r), (-e, e)],
        [(-e, -e), (-r, -r), (-r, r)],
        # East strip
        [(e, -e), (e, e), (r, r)],
        [(e, -e), (r, r), (r, -r)],
    ]


def _build_sea_node(sea_z: float, extent: float,
                    sea_polygon=None, radius_m: float | None = None) -> GeomNode:
    """Build the sea plane geometry.

    Within the terrain bbox the mesh follows the sea polygon exactly.
    Outside the terrain bbox it fills the full extent with 8 hardcoded
    triangles (no Shapely needed there).  A DepthOffsetAttrib shifts the
    sea slightly toward the camera in the depth buffer to eliminate the last
    bit of Z-fighting at the coastline edge.
    """
    if sea_polygon is not None and not sea_polygon.is_empty and radius_m is not None:
        sea_tris = (
            _sea_tris_outer(radius_m, extent) +
            _sea_tris_inner(sea_polygon, radius_m)
        )
    else:
        e = extent
        sea_tris = [
            [(-e, -e), (e, -e), (e, e)],
            [(-e, -e), (e, e), (-e, e)],
        ]

    vfmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData("sea", vfmt, Geom.UHStatic)
    vdata.setNumRows(len(sea_tris) * 3)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")

    prim = GeomTriangles(Geom.UHStatic)
    vi = 0
    for tri_coords in sea_tris:
        for (x, y) in tri_coords:
            vw.addData3(float(x), float(y), float(sea_z))
            nw.addData3(0.0, 0.0, 1.0)
        prim.addVertices(vi, vi + 1, vi + 2)
        vi += 3
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("sea")
    node.addGeom(geom)
    # Shift the sea plane slightly toward the camera in the depth buffer so it
    # always wins against terrain vertices that are at exactly the same z.
    node.setAttrib(DepthOffsetAttrib.make(1))
    return node


def _build_water_poly_node(water: WaterData) -> GeomNode | None:
    """Flat polygon mesh for lakes, ponds, and river-bank areas."""
    if not water.lake_polygons:
        return None
    vfmt = GeomVertexFormat.getV3n3()
    vdata = GeomVertexData("water_poly", vfmt, Geom.UHStatic)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    prim = GeomTriangles(Geom.UHStatic)
    vi = 0
    for poly, surface_z in water.lake_polygons:
        # Proportional max_seg: fine for small ponds, coarser for big lakes.
        max_seg = max(30.0, poly.area ** 0.5 / 20.0)
        for tri_coords in _triangulate_flat_poly(poly, max_seg):
            for (x, y) in tri_coords:
                vw.addData3(float(x), float(y), float(surface_z))
                nw.addData3(0.0, 0.0, 1.0)
            prim.addVertices(vi, vi + 1, vi + 2)
            vi += 3
    if vi == 0:
        return None
    vdata.setNumRows(vi)
    prim.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("water_poly")
    node.addGeom(geom)
    node.setAttrib(DepthOffsetAttrib.make(2))
    return node


def _build_water_river_node(water: WaterData) -> GeomNode | None:
    """Terrain-following polylines for rivers, streams, and canals."""
    if not water.rivers:
        return None
    ls = LineSegs("rivers")
    ls.setColor(0.15, 0.45, 0.75, 1.0)
    ls.setThickness(2.0)
    for river in water.rivers:
        ls.moveTo(float(river[0, 0]), float(river[0, 1]), float(river[0, 2]))
        for pt in river[1:]:
            ls.drawTo(float(pt[0]), float(pt[1]), float(pt[2]))
    return ls.create()


def _build_coastline_node(coast: CoastlineData, z: float) -> GeomNode | None:
    if not coast.polylines:
        return None
    ls = LineSegs("coastline")
    ls.setColor(1.0, 0.85, 0.2, 1.0)
    ls.setThickness(2.0)
    for poly in coast.polylines:
        ls.moveTo(float(poly[0, 0]), float(poly[0, 1]), z)
        for p in poly[1:]:
            ls.drawTo(float(p[0]), float(p[1]), z)
    return ls.create()


class TerrainViewer(ShowBase):
    MOVE_KEYS = ("w", "a", "s", "d", "q", "e")

    def __init__(self, terrain: TerrainData,
                 coastline: CoastlineData | None = None,
                 sea_z: float | None = None,
                 sea_polygon=None,
                 water: WaterData | None = None):
        ShowBase.__init__(self)

        props = WindowProperties()
        props.setTitle("osm3denv — terrain")
        self.win.requestProperties(props)

        self.setBackgroundColor(0.53, 0.70, 0.86, 1.0)

        terrain_node = _build_terrain_node(terrain)
        terrain_np = self.render.attachNewNode(terrain_node)
        terrain_np.setColor(Vec4(0.45, 0.55, 0.35, 1.0))

        r = float(terrain.radius_m)

        if water is not None:
            water_poly_node = _build_water_poly_node(water)
            if water_poly_node is not None:
                water_poly_np = self.render.attachNewNode(water_poly_node)
                water_poly_np.setColor(Vec4(0.15, 0.40, 0.65, 1.0))
                water_poly_np.setTwoSided(True)
            water_river_node = _build_water_river_node(water)
            if water_river_node is not None:
                water_river_np = self.render.attachNewNode(water_river_node)
                water_river_np.setLightOff()

        if sea_z is not None:
            sea_np = self.render.attachNewNode(
                _build_sea_node(sea_z, extent=r * 20.0,
                                sea_polygon=sea_polygon, radius_m=r))
            sea_np.setColor(Vec4(0.08, 0.25, 0.40, 1.0))
            sea_np.setTwoSided(True)

            if coastline is not None:
                coast_node = _build_coastline_node(coastline, z=sea_z + 0.5)
                if coast_node is not None:
                    coast_np = self.render.attachNewNode(coast_node)
                    coast_np.setLightOff()

        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.35, 0.35, 0.40, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.95, 0.92, 0.85, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-30, -50, 0)
        self.render.setLight(sun_np)

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
               coastline: CoastlineData | None = None,
               sea_z: float | None = None,
               sea_polygon=None,
               water: WaterData | None = None) -> None:
    TerrainViewer(terrain, coastline=coastline, sea_z=sea_z,
                  sea_polygon=sea_polygon, water=water).run()
