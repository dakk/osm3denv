"""OgreBites application context for the osm3denv viewer."""
from __future__ import annotations

import logging

import Ogre
import Ogre.Bites as OB
import Ogre.RTShader

from osm3denv.render import materials, upload
from osm3denv.render.camera import FreeCamera

log = logging.getLogger(__name__)


class ViewerApp(OB.ApplicationContext, OB.InputListener):
    def __init__(self, terrain, buildings, roads, water):
        OB.ApplicationContext.__init__(self, "osm3denv")
        OB.InputListener.__init__(self)
        self._terrain = terrain
        self._buildings = buildings
        self._roads = roads
        self._water = water
        self._camera: FreeCamera | None = None

    # ApplicationContext.setup is called after initApp() creates the window + root.
    def setup(self):
        OB.ApplicationContext.setup(self)
        self.addInputListener(self)

        root = self.getRoot()
        scn = root.createSceneManager()
        Ogre.RTShader.ShaderGenerator.getSingleton().addSceneManager(scn)

        scn.setAmbientLight(Ogre.ColourValue(0.35, 0.35, 0.40))
        sun = scn.createLight("sun")
        sun.setType(Ogre.Light.LT_DIRECTIONAL)
        sun.setDiffuseColour(Ogre.ColourValue(1.0, 0.95, 0.85))
        sun.setSpecularColour(Ogre.ColourValue(0.3, 0.3, 0.3))
        sun_node = scn.getRootSceneNode().createChildSceneNode()
        sun_node.setDirection(-0.4, -0.8, -0.4, Ogre.Node.TS_WORLD)
        sun_node.attachObject(sun)

        cam = scn.createCamera("cam")
        cam.setNearClipDistance(1.0)
        # Far plane sized to the scene; tighter = more depth precision near the ground.
        cam.setFarClipDistance(max(3000.0, 3.0 * float(self._terrain.radius_m)))
        cam.setAutoAspectRatio(True)
        cam_node = scn.getRootSceneNode().createChildSceneNode()
        cam_node.attachObject(cam)

        # Spawn slightly above terrain at the origin, looking north (−z in our mapping).
        y0 = float(self._terrain.sampler.height_at(0.0, 0.0)) + 1.7
        cam_node.setPosition(0.0, y0, 0.0)
        cam_node.lookAt(Ogre.Vector3(0.0, y0, -100.0), Ogre.Node.TS_WORLD)

        vp = self.getRenderWindow().addViewport(cam)
        vp.setBackgroundColour(Ogre.ColourValue(0.6, 0.75, 0.90))

        self._build_scene(scn)

        self._camera = FreeCamera(node=cam_node, walk_speed=10.0)
        self.addInputListener(self._camera)
        # Capture the mouse so rotation can sweep past screen edges.
        self.setWindowGrab(True)

    def _build_scene(self, scn):
        t = self._terrain
        upload.attach(scn, "terrain", t.vertices, t.normals, t.indices, materials.terrain())
        if self._buildings is not None:
            b = self._buildings
            upload.attach(scn, "buildings", b.vertices, b.normals, b.indices, materials.buildings())
        if self._roads is not None:
            r = self._roads
            upload.attach(scn, "roads", r.vertices, r.normals, r.indices, materials.roads())
        if self._water is not None:
            w = self._water
            upload.attach(scn, "water", w.vertices, w.normals, w.indices, materials.water())

    def keyPressed(self, evt) -> bool:
        if evt.keysym.sym == OB.SDLK_ESCAPE:
            self.getRoot().queueEndRendering()
            return True
        return False


def run_viewer(*, terrain, buildings, roads, water) -> None:
    app = ViewerApp(terrain, buildings, roads, water)
    app.initApp()
    try:
        app.getRoot().startRendering()
    finally:
        app.closeApp()
