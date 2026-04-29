"""Shared Panda3D helpers used by MapEntity.attach_to() implementations."""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def tod_intensity(time_of_day: float) -> float:
    """Night-time light intensity in [0,1]: 1.0 at midnight, 0.0 around noon."""
    sin_el = -math.cos(2.0 * math.pi * time_of_day)
    return float(np.clip((-sin_el + 0.05) / 0.15, 0.0, 1.0))


def nearest_k_idx(pos2d: np.ndarray, cam_e: float, cam_n: float, k: int) -> np.ndarray:
    """Indices of the *k* rows in *pos2d* closest to (cam_e, cam_n).

    When len(pos2d) <= k every index is returned.  Callers that hit this path
    every frame should cache ``np.arange(len(pos2d))`` and pass it directly
    rather than calling this function.
    """
    n_all = len(pos2d)
    k     = min(k, n_all)
    diffs = pos2d - np.array([cam_e, cam_n], dtype=np.float32)
    dists = (diffs * diffs).sum(axis=1)
    return np.argpartition(dists, k - 1)[:k] if k < n_all else np.arange(n_all)


_SHADER_DIR = Path(__file__).parent / "shaders"
_shader_cache: dict = {}


def load_shader(name: str):
    if name in _shader_cache:
        return _shader_cache[name]
    from panda3d.core import Shader
    vert = _SHADER_DIR / f"{name}.vert"
    frag = _SHADER_DIR / f"{name}.frag"
    if not vert.exists() or not frag.exists():
        log.warning("shader '%s' not found in %s", name, _SHADER_DIR)
        return None
    shader = Shader.load(Shader.SL_GLSL, vertex=str(vert), fragment=str(frag))
    _shader_cache[name] = shader
    return shader


def attach_mesh(parent, name: str,
                vertices: np.ndarray, normals: np.ndarray,
                uvs: np.ndarray | None = None,
                indices: np.ndarray | None = None,
                depth_offset: int = 0):
    """Build a triangle-mesh GeomNode and attach it to *parent*.

    Returns the resulting NodePath so the caller can set shaders, colors, etc.
    """
    from panda3d.core import (
        DepthOffsetAttrib, Geom, GeomNode, GeomTriangles,
        GeomVertexData, GeomVertexFormat, GeomVertexWriter,
    )
    has_uvs = uvs is not None
    vfmt = GeomVertexFormat.getV3n3t2() if has_uvs else GeomVertexFormat.getV3n3()
    vdata = GeomVertexData(name, vfmt, Geom.UHStatic)
    n = len(vertices)
    vdata.setNumRows(n)

    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    if has_uvs:
        tw = GeomVertexWriter(vdata, "texcoord")
    for i in range(n):
        vw.addData3(float(vertices[i, 0]), float(vertices[i, 1]), float(vertices[i, 2]))
        nw.addData3(float(normals[i, 0]),  float(normals[i, 1]),  float(normals[i, 2]))
        if has_uvs:
            tw.addData2(float(uvs[i, 0]), float(uvs[i, 1]))

    prim = GeomTriangles(Geom.UHStatic)
    if indices is not None:
        idx = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
        for a, b, c in idx:
            prim.addVertices(int(a), int(b), int(c))
    else:
        for i in range(0, n, 3):
            prim.addVertices(i, i + 1, i + 2)
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    if depth_offset:
        node.setAttrib(DepthOffsetAttrib.make(depth_offset))
    return parent.attachNewNode(node)


def attach_lines(parent, name: str,
                 polylines: list[np.ndarray],
                 color: tuple,
                 thickness: float = 2.0):
    """Build a LineSegs node and attach it to *parent*. Returns the NodePath."""
    from panda3d.core import LineSegs
    ls = LineSegs(name)
    ls.setColor(*color)
    ls.setThickness(thickness)
    for poly in polylines:
        if len(poly) < 2:
            continue
        ls.moveTo(float(poly[0, 0]), float(poly[0, 1]), float(poly[0, 2]))
        for pt in poly[1:]:
            ls.drawTo(float(pt[0]), float(pt[1]), float(pt[2]))
    return parent.attachNewNode(ls.create())
