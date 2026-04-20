"""Upload numpy vertex/index arrays to Ogre as a ManualObject."""
from __future__ import annotations

import numpy as np

import Ogre


def attach(scene, name: str, vertices, normals, indices, material_name: str,
           uvs=None):
    """Create a ManualObject and attach it to the scene root.

    ``uvs`` is an optional (N, 2) float32 array of texture coordinates.
    Returns the SceneNode, or None if geometry is empty.
    """
    if vertices is None or len(vertices) == 0 or len(indices) == 0:
        return None

    mo = scene.createManualObject(name)
    mo.begin(material_name, Ogre.RenderOperation.OT_TRIANGLE_LIST)

    verts = np.asarray(vertices, dtype=np.float32)
    norms = np.asarray(normals, dtype=np.float32)
    idx = np.asarray(indices, dtype=np.uint32).ravel()
    have_uv = uvs is not None
    if have_uv:
        uv_arr = np.asarray(uvs, dtype=np.float32)
        if uv_arr.shape[0] != verts.shape[0]:
            raise ValueError(f"{name}: uv count {uv_arr.shape[0]} != vertex count {verts.shape[0]}")

    for i in range(len(verts)):
        mo.position(float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2]))
        mo.normal(float(norms[i, 0]), float(norms[i, 1]), float(norms[i, 2]))
        if have_uv:
            mo.textureCoord(float(uv_arr[i, 0]), float(uv_arr[i, 1]))
    for i in range(0, len(idx), 3):
        mo.triangle(int(idx[i]), int(idx[i + 1]), int(idx[i + 2]))

    mo.end()
    node = scene.getRootSceneNode().createChildSceneNode()
    node.attachObject(mo)
    return node
