"""Renderer-agnostic description of a single displayable scene layer.

A ``RenderLayer`` carries geometry data (triangle mesh and/or polylines) plus
rendering hints (colour, depth offset, lighting).  Mesh modules produce layers;
the renderer consumes them without needing to know what they represent.

Geometry modes (may be combined in one layer):
* **Triangle mesh** — *vertices* + *normals* (+ optional *uvs* + optional
  *indices*).  When *indices* is ``None`` the vertex array is a flat triangle
  soup (len(vertices) == T × 3).
* **Polylines** — list of (K, 3) float32 arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RenderLayer:
    name: str

    # --- triangle mesh ---
    vertices: np.ndarray | None = None  # (N, 3) float32
    normals:  np.ndarray | None = None  # (N, 3) float32
    uvs:      np.ndarray | None = None  # (N, 2) float32 — optional
    indices:  np.ndarray | None = None  # (M,)  uint32; None → triangle soup

    # --- polylines ---
    polylines:      list[np.ndarray] | None = None  # each (K, 3) float32
    line_thickness: float = 2.0

    # --- rendering hints ---
    color:        tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    depth_offset: int = 0
    lit:          bool = True
    two_sided:    bool = False
    shader_name:   str | None = None   # named GLSL shader from render/shaders/
    shader_inputs: dict = field(default_factory=dict)  # uniform name → value
