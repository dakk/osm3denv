"""Upload numpy vertex/index arrays to Ogre as a ManualObject."""
from __future__ import annotations

import numpy as np

import Ogre


def attach(scene, name: str, vertices, normals, indices, material_name: str):
    """Create a ManualObject with the given geometry and attach it to the scene root.

    Returns the SceneNode holding the object, or None if geometry is empty.
    """
    if vertices is None or len(vertices) == 0 or len(indices) == 0:
        return None

    mo = scene.createManualObject(name)
    mo.begin(material_name, Ogre.RenderOperation.OT_TRIANGLE_LIST)

    verts = np.asarray(vertices, dtype=np.float32)
    norms = np.asarray(normals, dtype=np.float32)
    idx = np.asarray(indices, dtype=np.uint32).ravel()

    for i in range(len(verts)):
        mo.position(float(verts[i, 0]), float(verts[i, 1]), float(verts[i, 2]))
        mo.normal(float(norms[i, 0]), float(norms[i, 1]), float(norms[i, 2]))
    for i in range(0, len(idx), 3):
        mo.triangle(int(idx[i]), int(idx[i + 1]), int(idx[i + 2]))

    mo.end()
    node = scene.getRootSceneNode().createChildSceneNode()
    node.attachObject(mo)
    return node
