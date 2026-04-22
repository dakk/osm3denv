"""First-person camera controller.

Rotates on any mouse motion (no button hold), WASD+QE/Space+Ctrl movement,
Shift sprint. Assumes the window has grabbed the mouse so rotation can sweep
freely past the screen edges.
"""
from __future__ import annotations

import math

import Ogre
import Ogre.Bites as OB

# ogre-python exposes only a subset of SDL keycodes as attributes. Fill in the
# missing ones with the raw SDL2 values. Letter keys use the ASCII codepoint.
_K_W = ord("w")
_K_A = ord("a")
_K_S = ord("s")
_K_D = ord("d")
_K_E = ord("e")
_K_Q = ord("q")
_K_RSHIFT = 1073742053  # SDL_SCANCODE_TO_KEYCODE(SDL_SCANCODE_RSHIFT)
_K_LCTRL = 1073742048
_K_RCTRL = 1073742052


class FreeCamera(OB.InputListener):
    def __init__(self, node: Ogre.SceneNode, *,
                 walk_speed: float = 10.0,
                 sprint_mult: float = 5.0,
                 mouse_sensitivity: float = 0.15):
        OB.InputListener.__init__(self)
        self.node = node
        self.walk_speed = walk_speed
        self.sprint_mult = sprint_mult
        self.mouse_sensitivity = mouse_sensitivity
        self._keys: set[int] = set()
        self._yaw_deg = 0.0
        self._pitch_deg = 0.0
        self._init_from_node()

    def _init_from_node(self) -> None:
        q = self.node.getOrientation()
        forward = q * Ogre.Vector3(0, 0, -1)
        self._yaw_deg = math.degrees(math.atan2(forward.x, -forward.z))
        self._pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, forward.y))))

    def keyPressed(self, evt) -> bool:
        self._keys.add(evt.keysym.sym)
        return False

    def keyReleased(self, evt) -> bool:
        self._keys.discard(evt.keysym.sym)
        return False

    def mouseMoved(self, evt) -> bool:
        self._yaw_deg -= evt.xrel * self.mouse_sensitivity
        self._pitch_deg -= evt.yrel * self.mouse_sensitivity
        self._pitch_deg = max(-89.0, min(89.0, self._pitch_deg))
        self._apply_orientation()
        return True

    def frameRendered(self, evt) -> bool:
        dt = float(evt.timeSinceLastFrame)
        speed = self.walk_speed
        if OB.SDLK_LSHIFT in self._keys or _K_RSHIFT in self._keys:
            speed *= self.sprint_mult

        yaw = math.radians(self._yaw_deg)
        # Yaw-only frame so WASD is level regardless of pitch.
        # For a Ogre camera looking down -Z at yaw=0: forward=(0,0,-1), right=(1,0,0).
        # Rotating about +Y by yaw θ: forward=(-sin θ,0,-cos θ), right=(cos θ,0,-sin θ).
        fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
        right_x, right_z = math.cos(yaw), -math.sin(yaw)

        dx = dy = dz = 0.0
        if _K_W in self._keys: dx += fwd_x;   dz += fwd_z
        if _K_S in self._keys: dx -= fwd_x;   dz -= fwd_z
        if _K_D in self._keys: dx += right_x; dz += right_z
        if _K_A in self._keys: dx -= right_x; dz -= right_z
        if OB.SDLK_SPACE in self._keys or _K_E in self._keys: dy += 1.0
        if _K_LCTRL in self._keys or _K_RCTRL in self._keys or _K_Q in self._keys: dy -= 1.0

        if dx or dy or dz:
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            scale = speed * dt / length
            self.node.translate(Ogre.Vector3(dx * scale, dy * scale, dz * scale),
                                Ogre.Node.TS_WORLD)
        return True

    def _apply_orientation(self) -> None:
        yaw_q = Ogre.Quaternion(Ogre.Degree(self._yaw_deg), Ogre.Vector3(0, 1, 0))
        pitch_q = Ogre.Quaternion(Ogre.Degree(self._pitch_deg), Ogre.Vector3(1, 0, 0))
        self.node.setOrientation(yaw_q * pitch_q)
