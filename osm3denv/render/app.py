"""OgreBites application context for the osm3denv viewer."""
from __future__ import annotations

import logging
from pathlib import Path

import Ogre
import Ogre.Bites as OB
import Ogre.RTShader

from osm3denv.render import materials, plants as plants_mod, upload
from osm3denv.render.camera import FreeCamera
from osm3denv.render.sky import SkyFollower, SunController, build_sky_dome

log = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


class ViewerApp(OB.ApplicationContext, OB.InputListener):
    def __init__(self, terrain, buildings, roads, water, areas, trees,
                 *, furniture=None, texture_root: Path | None = None,
                 plant_pack_root: Path | None = None):
        OB.ApplicationContext.__init__(self, "osm3denv")
        OB.InputListener.__init__(self)
        self._terrain = terrain
        self._buildings = buildings
        self._roads = roads
        self._water = water
        self._areas = areas or []
        self._trees = trees
        self._furniture = furniture or []
        self._texture_root = texture_root
        self._plant_pack_root = plant_pack_root
        self._plants = None  # populated in setup() once Ogre is initialised
        self._camera: FreeCamera | None = None
        # Let the materials factories know whether PBR textures are cached.
        materials.set_texture_root(texture_root)

    def locateResources(self):
        # Inject our asset directories before Ogre loads resources.
        OB.ApplicationContext.locateResources(self)
        rgm = Ogre.ResourceGroupManager.getSingleton()
        for sub in ("shaders", "shaders/postfx", "materials", "compositors"):
            p = _ASSETS_DIR / sub
            if p.is_dir():
                rgm.addResourceLocation(str(p), "FileSystem",
                                        Ogre.RGN_DEFAULT, True)
                log.debug("resource location added: %s", p)
        # PBR texture packs cached by osm3denv.fetch.textures. Add each pack
        # subdirectory directly — Ogre's recursive lookup on the cache root
        # doesn't surface files in subfolders for resource-by-name queries.
        if self._texture_root is not None and self._texture_root.is_dir():
            for pack_dir in sorted(self._texture_root.iterdir()):
                if pack_dir.is_dir():
                    rgm.addResourceLocation(str(pack_dir), "FileSystem",
                                            Ogre.RGN_DEFAULT, False)
                    log.debug("resource location added: %s", pack_dir)
        # 3D mesh packs (Shapespark plants kit). Register the pack root +
        # every subdirectory so Assimp can find the glTF and its companion
        # buffer/textures next to it.
        if self._plant_pack_root is not None and self._plant_pack_root.is_dir():
            for p in [self._plant_pack_root, *[d for d in self._plant_pack_root.rglob('*') if d.is_dir()]]:
                rgm.addResourceLocation(str(p), "FileSystem",
                                        Ogre.RGN_DEFAULT, False)
                log.debug("resource location added: %s", p)

    # ApplicationContext.setup is called after initApp() creates the window + root.
    def setup(self):
        OB.ApplicationContext.setup(self)
        self.addInputListener(self)

        root = self.getRoot()
        scn = root.createSceneManager()
        Ogre.RTShader.ShaderGenerator.getSingleton().addSceneManager(scn)

        scn.setAmbientLight(Ogre.ColourValue(0.35, 0.35, 0.40))

        # Directional shadow map for the sun. Modulative mode applies shadow
        # darkening as a separate post pass, so it doesn't require changes to
        # our custom surface shaders. FocusedShadowCameraSetup fits the shadow
        # camera tightly to the view frustum for the best shadow resolution.
        scn.setShadowTechnique(Ogre.SHADOWTYPE_TEXTURE_MODULATIVE)
        scn.setShadowTextureCount(1)
        scn.setShadowTextureSize(2048)
        scn.setShadowTexturePixelFormat(Ogre.PF_DEPTH16)
        scn.setShadowFarDistance(min(3000.0, 2.5 * float(self._terrain.radius_m)))
        scn.setShadowColour(Ogre.ColourValue(0.45, 0.45, 0.52))
        scn.setShadowDirectionalLightExtrusionDistance(10000.0)
        # .create() on the concrete Ptr returns a ShadowCameraSetupPtr that
        # setShadowCameraSetup accepts; passing a plain FocusedShadowCameraSetup
        # fails SWIG's Ptr typecheck.
        scn.setShadowCameraSetup(Ogre.FocusedShadowCameraSetupPtr().create())

        sun = scn.createLight("sun")
        sun.setType(Ogre.Light.LT_DIRECTIONAL)
        sun.setCastShadows(True)
        sun_node = scn.getRootSceneNode().createChildSceneNode()
        sun_node.attachObject(sun)
        # Initial direction; SunController will overwrite it based on time of day.
        sun_node.setDirection(-0.4, -0.8, -0.4, Ogre.Node.TS_WORLD)

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
        # Background is a fallback; the sky dome covers the viewport every frame.
        vp.setBackgroundColour(Ogre.ColourValue(0.0, 0.0, 0.0))

        # HDR + bloom + ACES tonemap compositor. Attaching the compositor to
        # the viewport routes the scene through our post-fx chain.
        self._enable_postfx(vp)

        # Pre-load + slice the Shapespark plants kit so Entities created in
        # _build_scene can reference per-plant meshes by name.
        from osm3denv.fetch import meshes as _mesh_fetch
        gltf_path = None
        if self._plant_pack_root is not None:
            # derive from fetch config so the path logic stays in one place
            info = _mesh_fetch.MESH_PACKS["shapespark_plants"]
            candidate = self._plant_pack_root / info["gltf"]
            if candidate.is_file():
                gltf_path = candidate
        if gltf_path is not None:
            self._plants = plants_mod.load_kit(gltf_path)
        else:
            self._plants = plants_mod.PlantKit()

        self._build_scene(scn)

        sky_node = build_sky_dome(scn, cam_node)
        self._sky_follower = SkyFollower(sky_node, cam_node)
        self.addInputListener(self._sky_follower)

        self._sun_controller = SunController(sun_node, sun, initial_hour=10.0)
        self.addInputListener(self._sun_controller)

        self._camera = FreeCamera(node=cam_node, walk_speed=10.0)
        self.addInputListener(self._camera)
        # Capture the mouse so rotation can sweep past screen edges.
        self.setWindowGrab(True)

    def _enable_postfx(self, vp: "Ogre.Viewport") -> None:
        cm = Ogre.CompositorManager.getSingleton()
        inst = cm.addCompositor(vp, "osm3denv/postfx")
        if inst is None:
            log.warning("postfx compositor could not be added; check ogre.log")
            return
        cm.setCompositorEnabled(vp, "osm3denv/postfx", True)

        # Feed the blur passes a texel_size matching their quarter-res RT so
        # the sample offsets actually land one pixel apart.
        w = max(1, int(self.getRenderWindow().getWidth()))
        h = max(1, int(self.getRenderWindow().getHeight()))
        step = Ogre.Vector2(1.0 / (w * 0.25), 1.0 / (h * 0.25))
        for mat_name in ("osm3d/postfx/blur_h", "osm3d/postfx/blur_v"):
            mat = Ogre.MaterialManager.getSingleton().getByName(mat_name)
            if mat is None:
                continue
            params = mat.getTechnique(0).getPass(0).getFragmentProgramParameters()
            params.setNamedConstant("texel_size", step)

    def _build_scene(self, scn):
        t = self._terrain
        upload.attach(scn, "terrain", t.vertices, t.normals, t.indices,
                      materials.terrain(), uvs=t.uvs)
        # Area polygons (vegetation, residential, industrial, ...) follow the
        # terrain surface exactly; strong per-material depth_bias makes them
        # win the depth test against the terrain they sit on.
        for i, am in enumerate(self._areas):
            mat_name = am.material_factory()
            upload.attach(scn, f"area_{i}_{mat_name}",
                          am.vertices, am.normals, am.indices, mat_name,
                          uvs=am.uvs)
        if self._roads is not None:
            for r in self._roads:
                upload.attach(scn, f"roads_{r.kind}", r.vertices, r.normals,
                              r.indices, materials.roads_for_kind(r.kind),
                              uvs=r.uvs)
        if self._water is not None:
            w = self._water
            upload.attach(scn, "water", w.vertices, w.normals, w.indices,
                          materials.water(), uvs=w.uvs)
        if self._buildings is not None:
            for b in self._buildings:
                upload.attach(scn, f"buildings_v{b.variant}",
                              b.vertices, b.normals, b.indices,
                              materials.buildings_for_variant(b.variant),
                              uvs=b.uvs, colors=b.colors)
                # Stone cornice trim rendered with a neutral warm-stone
                # material regardless of the building's wall pack.
                if len(b.trim_indices) > 0:
                    upload.attach(scn, f"buildings_trim_v{b.variant}",
                                  b.trim_vertices, b.trim_normals,
                                  b.trim_indices, materials.building_trim(),
                                  uvs=b.trim_uvs)
        if (self._trees is not None and self._trees.count > 0
                and self._plants and self._plants.num_trees > 0):
            self._attach_trees(scn)
        for fm in self._furniture:
            if fm.count == 0:
                continue
            mat = (materials.furniture_metal() if fm.kind == "lamp"
                   else materials.furniture_wood())
            upload.attach(scn, f"furniture_{fm.kind}",
                          fm.vertices, fm.normals, fm.indices, mat,
                          uvs=fm.uvs)

    def _attach_trees(self, scn) -> None:
        """Spawn one Entity per tree placement from the Shapespark plants kit.

        Species preference maps to a plant pick: conifer → plants with names
        containing "pine"/"spruce" when available, otherwise round-robin.
        Height is set by scaling the plant mesh vertically to match the
        OSM ``height`` tag (or the scatter default).
        """
        trees = self._plants.trees
        if not trees:
            return
        # Partition plants into conifer-looking and the rest based on glTF
        # node-name heuristics (Shapespark names include species hints).
        def _is_conifer(name: str) -> bool:
            n = name.lower()
            return any(k in n for k in ("pine", "spruce", "fir", "cedar",
                                        "conifer"))
        conifers = [p for p in trees if _is_conifer(p.gltf_name)]
        broadleaves = [p for p in trees if not _is_conifer(p.gltf_name)]
        if not conifers:
            conifers = trees
        if not broadleaves:
            broadleaves = trees

        root = scn.getRootSceneNode()
        y_axis = Ogre.Vector3(0.0, 1.0, 0.0)
        for i, tp in enumerate(self._trees.placements):
            pool = conifers if tp.species == "conifer" else broadleaves
            plant = pool[(tp.seed & 0x7FFFFFFF) % len(pool)]
            scale = tp.height / max(plant.height, 0.1)

            # The kit bakes a grid translation into each plant's vertex data
            # (e.g. x≈9.4, z≈2.7). To make the plant's centroid-XZ and base-Y
            # land at (east, base_y, -north) AFTER a yaw rotation, rotate the
            # pivot by the yaw quaternion first, then subtract scale*rotated
            # pivot from the target position.
            quat = Ogre.Quaternion(Ogre.Radian(float(tp.yaw_rad)), y_axis)
            rp = quat * Ogre.Vector3(plant.pivot_x, plant.pivot_y, plant.pivot_z)

            ent = scn.createEntity(f"tree_{i}", plant.name)
            node = root.createChildSceneNode()
            node.setPosition(tp.east - scale * rp.x,
                             tp.base_y - scale * rp.y,
                             -tp.north - scale * rp.z)
            node.setOrientation(quat)
            node.setScale(scale, scale, scale)
            node.attachObject(ent)

    def keyPressed(self, evt) -> bool:
        if evt.keysym.sym == OB.SDLK_ESCAPE:
            self.getRoot().queueEndRendering()
            return True
        return False


def run_viewer(*, terrain, buildings, roads, water, areas=None, trees=None,
               furniture=None,
               texture_root: Path | None = None,
               plant_pack_root: Path | None = None) -> None:
    app = ViewerApp(terrain, buildings, roads, water, areas, trees,
                    furniture=furniture,
                    texture_root=texture_root,
                    plant_pack_root=plant_pack_root)
    app.initApp()
    try:
        app.getRoot().startRendering()
    finally:
        app.closeApp()
