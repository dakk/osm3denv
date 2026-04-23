"""Sky-dome cloud layer — animated FBM clouds rendered on a hemisphere mesh."""
from __future__ import annotations

import builtins
import logging
import math

import numpy as np
from direct.task import Task

from osm3denv.entity import MapEntity

log = logging.getLogger(__name__)


class Clouds(MapEntity):
    SHADER = "clouds"

    def __init__(self, radius_m: float) -> None:
        self._radius_m = radius_m
        self._root = None

    def build(self) -> None:
        pass  # fully procedural

    def attach_to(self, parent) -> None:
        from panda3d.core import DepthTestAttrib, DepthWriteAttrib, TransparencyAttrib
        from osm3denv.render.helpers import attach_mesh, load_shader

        shader = load_shader(self.SHADER)
        if shader is None:
            log.warning("clouds shader not found; skipping clouds")
            return

        verts, normals, uvs, indices = self._build_dome()

        self._root = parent.attachNewNode("clouds_root")
        np_ = attach_mesh(self._root, "clouds_dome",
                          verts, normals, uvs=uvs, indices=indices)

        np_.setShader(shader)
        np_.setTransparency(TransparencyAttrib.M_alpha)
        np_.setAttrib(DepthTestAttrib.make(DepthTestAttrib.M_none))
        np_.setAttrib(DepthWriteAttrib.make(DepthWriteAttrib.M_off))
        np_.setBin("background", 1)
        np_.setLightOff()
        np_.setTwoSided(True)

        builtins.base.taskMgr.add(self._follow_camera, "clouds_follow")
        log.info("clouds attached (dome R=%.0f m)", self._dome_radius())

    # Keep the dome centred on the camera so its edge is never visible.
    def _follow_camera(self, task: Task.Task) -> int:
        if self._root is None or self._root.isEmpty():
            return Task.done
        pos = builtins.base.camera.getPos()
        self._root.setPos(pos.x, pos.y, 0.0)
        return Task.cont

    def _dome_radius(self) -> float:
        return float(np.clip(self._radius_m * 8.0, 30_000.0, 80_000.0))

    def _build_dome(self) -> tuple:
        """Hemisphere pointing up, interior-facing normals, UV=(az/2π, el/½π)."""
        N_phi   = 20   # elevation rings
        N_theta = 64   # azimuth segments
        R = self._dome_radius()

        verts, normals, uvs, indices = [], [], [], []

        for i in range(N_phi + 1):
            phi = i / N_phi * (math.pi / 2.0)
            cp, sp = math.cos(phi), math.sin(phi)
            for j in range(N_theta + 1):
                theta = j / N_theta * 2.0 * math.pi
                ct, st = math.cos(theta), math.sin(theta)
                verts.append([cp * ct * R,  cp * st * R,  sp * R])
                normals.append([-cp * ct,   -cp * st,     -sp])
                uvs.append([j / N_theta,    i / N_phi])

        for i in range(N_phi):
            for j in range(N_theta):
                a = i * (N_theta + 1) + j
                b, c, d = a + 1, a + (N_theta + 1), a + (N_theta + 2)
                indices.extend([a, c, b, b, c, d])

        return (np.array(verts,   dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(uvs,     dtype=np.float32),
                np.array(indices, dtype=np.uint32))
