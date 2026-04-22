"""Atmospheric sky dome + time-of-day sun controller.

The dome is an inverted unit cube attached to a scene node that tracks the
camera's *position* (but not its orientation). The shader in ``sky.frag`` uses
each vertex's object-local position as the view direction, which is why the
node must not inherit camera rotation.

The :class:`SunController` rotates the directional sun light in the scene; the
sky shader reads this direction via Ogre's ``light_direction`` auto-param.
"""
from __future__ import annotations

import math

import logging

import Ogre
import Ogre.Bites as OB

log = logging.getLogger(__name__)


def build_sky_dome(scn: "Ogre.SceneManager",
                   camera_node: "Ogre.SceneNode") -> "Ogre.SceneNode":
    """Create the sky dome and attach it so it follows the camera.

    Returns the sky scene node in case the caller wants to toggle visibility.
    """
    mo = scn.createManualObject("sky_dome")
    mo.setCastShadows(False)
    # Render before everything else, with depth writes disabled.
    mo.setRenderQueueGroup(Ogre.RENDER_QUEUE_SKIES_EARLY)

    mo.begin("osm3d/sky", Ogre.RenderOperation.OT_TRIANGLE_LIST)

    # Inverted cube. Size only has to clear the camera near plane (1.0 m here);
    # sky.vert then forces every fragment to the far plane. Winding doesn't
    # matter because the material uses CULL_NONE.
    s = 10.0
    corners = [
        (-s, -s, -s), (+s, -s, -s), (+s, +s, -s), (-s, +s, -s),
        (-s, -s, +s), (+s, -s, +s), (+s, +s, +s), (-s, +s, +s),
    ]
    for x, y, z in corners:
        mo.position(x, y, z)

    # 12 triangles (2 per face), wound outward (doesn't matter with CULL_NONE).
    faces = [
        (0, 1, 2), (0, 2, 3),   # -Z
        (5, 4, 7), (5, 7, 6),   # +Z
        (4, 0, 3), (4, 3, 7),   # -X
        (1, 5, 6), (1, 6, 2),   # +X
        (3, 2, 6), (3, 6, 7),   # +Y
        (4, 5, 1), (4, 1, 0),   # -Y
    ]
    for a, b, c in faces:
        mo.triangle(a, b, c)
    mo.end()

    # Infinite AABB so frustum culling never drops the dome.
    mo.setBoundingBox(Ogre.AxisAlignedBox.BOX_INFINITE)

    sky_node = scn.getRootSceneNode().createChildSceneNode("sky_node")
    sky_node.attachObject(mo)
    return sky_node


class SkyFollower(OB.InputListener):
    """Per-frame listener that glues the sky node to the camera's position."""

    def __init__(self, sky_node: "Ogre.SceneNode",
                 camera_node: "Ogre.SceneNode") -> None:
        OB.InputListener.__init__(self)
        self._sky_node = sky_node
        self._camera_node = camera_node

    def frameRendered(self, evt) -> bool:
        self._sky_node.setPosition(self._camera_node.getPosition())
        return True


class SunController(OB.InputListener):
    """Rotates the sun light based on a mutable time-of-day.

    ``time_of_day`` is in hours on a 0–24 cycle. The sun traces a simple arc:
    rises at 06:00 from the east, noon at 12:00 at the zenith, sets at 18:00
    to the west. Keys: ``T`` = +1h, ``[`` / ``]`` = -30 min / +30 min.
    Time changes are logged to stdout.
    """

    def __init__(self, sun_node: "Ogre.SceneNode", sun_light: "Ogre.Light",
                 *, initial_hour: float = 10.0,
                 step_hours_per_second: float = 0.0) -> None:
        OB.InputListener.__init__(self)
        self._sun_node = sun_node
        self._sun_light = sun_light
        self._hour = float(initial_hour) % 24.0
        self._step_rate = float(step_hours_per_second)
        self._keys: set[int] = set()
        self._apply()

    @property
    def hour(self) -> float:
        return self._hour

    @hour.setter
    def hour(self, v: float) -> None:
        self._hour = float(v) % 24.0
        self._apply()

    def keyPressed(self, evt) -> bool:
        self._keys.add(evt.keysym.sym)
        sym = evt.keysym.sym
        if sym == ord("["):
            self.hour = self._hour - 0.5
        elif sym == ord("]"):
            self.hour = self._hour + 0.5
        elif sym == ord("t"):
            self.hour = self._hour + 1.0
        return False

    def keyReleased(self, evt) -> bool:
        self._keys.discard(evt.keysym.sym)
        return False

    def frameRendered(self, evt) -> bool:
        if self._step_rate != 0.0:
            self.hour = self._hour + self._step_rate * float(evt.timeSinceLastFrame)
        return True

    def _apply(self) -> None:
        h = int(self._hour) % 24
        m = int(round((self._hour - int(self._hour)) * 60.0)) % 60
        print(f"time {h:02d}:{m:02d}", flush=True)

        # Sun elevation traces a sine: 0 at 06:00 and 18:00, 1 at 12:00, -1 at 00:00.
        phase = (self._hour - 6.0) / 12.0          # 0 at sunrise, 1 at sunset
        elev = math.sin(phase * math.pi)           # [-1..1]; negative before sunrise / after sunset
        # Azimuth: east at sunrise (-x), south at noon (-z in our mapping), west at sunset (+x).
        az = (self._hour / 24.0) * 2.0 * math.pi   # 0 at midnight
        # Build sun-TO direction vector (points at the sun).
        cos_e = math.cos(math.asin(max(-1.0, min(1.0, elev))))
        sx = -math.sin(az) * cos_e
        sz = -math.cos(az) * cos_e
        sy = elev
        # Light "direction" is the direction of propagation, i.e. FROM the sun
        # outward — negate the sun-TO vector.
        self._sun_node.setDirection(-sx, -sy, -sz, Ogre.Node.TS_WORLD)

        # Scale light intensity by elevation so dusk/night dim the scene.
        k = max(0.0, min(1.0, (elev + 0.1) / 0.4))
        warm = 1.0 - k                              # warm tint at dawn/dusk
        r = 1.0
        g = 0.95 - 0.25 * warm
        b = 0.85 - 0.45 * warm
        intensity = max(0.04, k)
        self._sun_light.setDiffuseColour(Ogre.ColourValue(r * intensity,
                                                          g * intensity,
                                                          b * intensity))
        spec = 0.3 * intensity
        self._sun_light.setSpecularColour(Ogre.ColourValue(spec, spec, spec))
