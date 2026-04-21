"""Load the Shapespark Low-Poly Plants Kit as one Ogre Mesh per plant.

The kit's monolithic glTF contains 30 plants, each glTF-mesh with ~2-3
primitives (trunk, foliage, flowers). Ogre's Assimp codec flattens the file
into a single Mesh with ~70 submeshes. We use :mod:`pygltflib` to read the
glTF node/mesh/primitive structure and determine which submesh indices belong
to which plant, then clone the Ogre Mesh per plant keeping *only* that
plant's submeshes so a single entity renders trunk + leaves together.

Plant species is classified by glTF node name prefix (``Tree-`` vs the rest)
and local-space height is computed from the per-plant cloned mesh bounds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import Ogre

log = logging.getLogger(__name__)


@dataclass
class Plant:
    name: str                # Ogre mesh name, e.g. "osm3d/plant_0"
    gltf_name: str           # original glTF node name, e.g. "Tree-01-1"
    kind: str                # "tree" or "bush"
    height: float


@dataclass
class PlantKit:
    trees: list[Plant] = field(default_factory=list)
    bushes: list[Plant] = field(default_factory=list)

    @property
    def num_trees(self) -> int:
        return len(self.trees)

    @property
    def num_bushes(self) -> int:
        return len(self.bushes)


def _read_gltf_plant_map(gltf_path: Path) -> list[tuple[str, list[int]]] | None:
    """Return [(node_name, [submesh indices])] based on glTF structure.

    Assimp preserves primitive order when flattening a glTF, so submesh
    indices are the cumulative primitive counts across meshes in glTF order.
    """
    try:
        from pygltflib import GLTF2
    except ImportError:
        log.warning("pygltflib missing; plant kit will not load correctly")
        return None
    try:
        g = GLTF2().load(str(gltf_path))
    except Exception as e:  # noqa: BLE001
        log.warning("failed to parse %s: %s", gltf_path, e)
        return None

    scene = g.scenes[g.scene or 0]
    # submesh_index starts at 0 and advances through primitives, walking the
    # scene's root nodes in order.
    out: list[tuple[str, list[int]]] = []
    next_submesh = 0

    def _collect(node_idx: int) -> list[int]:
        nonlocal next_submesh
        sub_indices: list[int] = []
        n = g.nodes[node_idx]
        if n.mesh is not None:
            m = g.meshes[n.mesh]
            for _ in m.primitives:
                sub_indices.append(next_submesh)
                next_submesh += 1
        for child in (n.children or []):
            sub_indices.extend(_collect(child))
        return sub_indices

    for root_idx in scene.nodes:
        name = g.nodes[root_idx].name or f"plant_{root_idx}"
        out.append((name, _collect(root_idx)))
    return out


def load_kit(gltf_path: Path,
             gltf_resource_name: str = "shapespark-low-poly-plants-kit.gltf"
             ) -> PlantKit:
    """Load + slice the kit via Ogre Assimp + glTF-structure mapping."""
    kit = PlantKit()
    plant_map = _read_gltf_plant_map(gltf_path)
    if not plant_map:
        return kit

    mm = Ogre.MeshManager.getSingleton()
    try:
        source = mm.load(gltf_resource_name, Ogre.RGN_DEFAULT)
    except Exception as e:  # noqa: BLE001
        log.warning("could not load plant glTF %s: %s", gltf_resource_name, e)
        return kit

    total_submeshes = source.getNumSubMeshes()
    expected = sum(len(s) for _, s in plant_map)
    if total_submeshes != expected:
        log.warning("plant submesh-count mismatch: ogre=%d gltf-derived=%d "
                    "(assimp reordered primitives); falling back to flat split",
                    total_submeshes, expected)
        # Fallback: one plant per submesh, no multi-primitive grouping.
        plant_map = [(f"plant_{i}", [i]) for i in range(total_submeshes)]

    log.info("plant kit: %d plants across %d submeshes", len(plant_map),
             total_submeshes)

    for plant_idx, (gltf_name, keep_indices) in enumerate(plant_map):
        clone_name = f"osm3d/plant_{plant_idx}"
        if mm.resourceExists(clone_name, Ogre.RGN_DEFAULT):
            clone = mm.getByName(clone_name, Ogre.RGN_DEFAULT)
        else:
            clone = source.clone(clone_name)
            keep = set(keep_indices)
            # Destroy back-to-front so earlier indices stay stable.
            for j in range(clone.getNumSubMeshes() - 1, -1, -1):
                if j not in keep:
                    clone.destroySubMesh(j)
            try:
                clone._updateBoundsFromVertexBuffers()
            except Exception:  # noqa: BLE001
                pass

        aabb = clone.getBounds()
        height = float(aabb.getMaximum().y - aabb.getMinimum().y)
        lower = gltf_name.lower()
        if lower.startswith("tree"):
            kind = "tree"
        elif height >= 3.0 and height <= 30.0:
            kind = "tree"
        else:
            kind = "bush"
        plant = Plant(name=clone_name, gltf_name=gltf_name, kind=kind,
                      height=height)
        if kind == "tree":
            kit.trees.append(plant)
        else:
            kit.bushes.append(plant)
        log.debug("plant %02d %r h=%.2f → %s (submeshes=%s)",
                  plant_idx, gltf_name, height, kind, keep_indices)

    log.info("plant kit classified: %d trees, %d bushes",
             kit.num_trees, kit.num_bushes)
    return kit
