from __future__ import annotations
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LPoint3f,
    LVector3f,
)

from procbuilding.params import Color

UV = tuple[float, float]


def face_normal(p0: LPoint3f, p1: LPoint3f, p2: LPoint3f) -> LVector3f:
    """Compute a normalised face normal from three CCW vertices."""
    a = p1 - p0
    b = p2 - p0
    n = a.cross(b)
    n.normalize()
    return n


class GeomBuilder:
    """
    Accumulates vertices and triangles then produces a GeomNode.

    Usage:
        b = GeomBuilder("wall")
        b.add_quad([p0, p1, p2, p3], normal, color, uvs=[(0,0),(1,0),(1,1),(0,1)])
        node = b.build()
        render.attachNewNode(node)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._verts: list[tuple[float, float, float]] = []
        self._normals: list[tuple[float, float, float]] = []
        self._colors: list[Color] = []
        self._uvs: list[UV] = []
        self._indices: list[tuple[int, int, int]] = []

    def add_quad(
        self,
        verts: list[LPoint3f],
        normal: LVector3f,
        color: Color,
        uvs: list[UV] | None = None,
    ) -> None:
        """Add a quad as two triangles. Vertices must be wound CCW from outside."""
        assert len(verts) == 4
        if uvs is None:
            uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert len(uvs) == 4
        base = len(self._verts)
        nx, ny, nz = normal.x, normal.y, normal.z
        for v, uv in zip(verts, uvs):
            self._verts.append((v.x, v.y, v.z))
            self._normals.append((nx, ny, nz))
            self._colors.append(color)
            self._uvs.append(uv)
        self._indices.append((base, base + 1, base + 2))
        self._indices.append((base, base + 2, base + 3))

    def add_triangle(
        self,
        verts: list[LPoint3f],
        normal: LVector3f,
        color: Color,
        uvs: list[UV] | None = None,
    ) -> None:
        """Add a single triangle. Vertices wound CCW from outside."""
        assert len(verts) == 3
        if uvs is None:
            uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        assert len(uvs) == 3
        base = len(self._verts)
        nx, ny, nz = normal.x, normal.y, normal.z
        for v, uv in zip(verts, uvs):
            self._verts.append((v.x, v.y, v.z))
            self._normals.append((nx, ny, nz))
            self._colors.append(color)
            self._uvs.append(uv)
        self._indices.append((base, base + 1, base + 2))

    def build(self) -> GeomNode:
        """Finalise and return a GeomNode ready to attach to the scene graph."""
        fmt = GeomVertexFormat.getV3n3c4t2()
        vdata = GeomVertexData(self._name, fmt, Geom.UHStatic)
        vdata.setNumRows(len(self._verts))

        v_writer = GeomVertexWriter(vdata, "vertex")
        n_writer = GeomVertexWriter(vdata, "normal")
        c_writer = GeomVertexWriter(vdata, "color")
        t_writer = GeomVertexWriter(vdata, "texcoord")

        for pos, norm, col, uv in zip(self._verts, self._normals, self._colors, self._uvs):
            v_writer.addData3f(*pos)
            n_writer.addData3f(*norm)
            c_writer.addData4f(*col)
            t_writer.addData2f(*uv)

        tris = GeomTriangles(Geom.UHStatic)
        for i0, i1, i2 in self._indices:
            tris.addVertices(i0, i1, i2)
        tris.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(tris)

        node = GeomNode(self._name)
        node.addGeom(geom)
        return node
