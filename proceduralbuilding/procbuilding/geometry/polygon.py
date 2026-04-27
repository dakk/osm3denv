"""2D polygon utilities: signed area, CCW normalization, ear-clip triangulation, L-shape preset."""
from __future__ import annotations

from typing import Sequence

Vert2D = tuple[float, float]


def signed_area(verts: Sequence[Vert2D]) -> float:
    """Shoelace formula. Positive → CCW (standard XY math orientation)."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area / 2.0


def ensure_ccw(verts: list[Vert2D]) -> list[Vert2D]:
    """Return verts wound CCW; reverses if signed_area is negative."""
    if signed_area(verts) < 0:
        return list(reversed(verts))
    return list(verts)


def _point_in_triangle(p: Vert2D, a: Vert2D, b: Vert2D, c: Vert2D) -> bool:
    """True if p is strictly inside triangle abc (barycentric test)."""
    def _sign(p1: Vert2D, p2: Vert2D, p3: Vert2D) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1, d2, d3 = _sign(p, a, b), _sign(p, b, c), _sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def ear_clip_triangulate(verts: Sequence[Vert2D]) -> list[tuple[Vert2D, Vert2D, Vert2D]]:
    """
    Triangulate a simple CCW polygon via ear clipping.

    Returns (p0, p1, p2) tuples wound CCW; suitable for add_triangle calls.
    Complexity is O(n³) — fine for small polygons (6-20 vertices).
    """
    poly = list(verts)
    n = len(poly)
    if n < 3:
        return []
    if n == 3:
        return [(poly[0], poly[1], poly[2])]

    tris: list[tuple[Vert2D, Vert2D, Vert2D]] = []
    remaining = list(range(n))

    while len(remaining) > 3:
        m = len(remaining)
        ear_found = False
        for i in range(m):
            ia = remaining[(i - 1) % m]
            ib = remaining[i]
            ic = remaining[(i + 1) % m]
            a, b, c = poly[ia], poly[ib], poly[ic]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 0:
                continue
            if not any(
                _point_in_triangle(poly[remaining[j]], a, b, c)
                for j in range(m)
                if j not in ((i - 1) % m, i, (i + 1) % m)
            ):
                tris.append((a, b, c))
                remaining.pop(i)
                ear_found = True
                break
        if not ear_found:
            break  # degenerate or self-intersecting polygon

    if len(remaining) == 3:
        tris.append((poly[remaining[0]], poly[remaining[1]], poly[remaining[2]]))
    return tris


def l_shape_verts(
    main_width: float,
    main_depth: float,
    notch_width: float,
    notch_depth: float,
) -> list[Vert2D]:
    """
    CCW vertices of an L-shaped footprint.

    Full bbox = main_width × main_depth.
    Notch of notch_width × notch_depth is cut from the back-right corner.

    Layout (Y increases northward, * = notch corner):

        (0, md)-----(mw-nw, md)
           |              |
           |         (mw-nw, md-nd)---(mw, md-nd)
           |                               |
        (0,  0)----------------(mw, 0)

    Vertices in CCW order starting at origin:
        (0,0) → (mw,0) → (mw, md-nd) → (mw-nw, md-nd) → (mw-nw, md) → (0, md)
    """
    mw, md, nw, nd = main_width, main_depth, notch_width, notch_depth
    return [
        (0.0,     0.0    ),
        (mw,      0.0    ),
        (mw,      md - nd),
        (mw - nw, md - nd),
        (mw - nw, md    ),
        (0.0,     md    ),
    ]
