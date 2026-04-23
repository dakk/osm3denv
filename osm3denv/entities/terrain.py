"""Terrain entity — heightmap mesh with elevation/slope shader."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from osm3denv.entity import MapEntity
from osm3denv.frame import Frame

log = logging.getLogger(__name__)


def _load_texture(path, fallback_rgb: bytes, *, srgb: bool):
    """Load a texture from *path*, or create a 1×1 solid fallback."""
    from panda3d.core import Filename, Texture
    tex = Texture()
    if path is not None:
        try:
            tex.read(Filename.fromOsSpecific(str(path)))
            tex.setMinfilter(Texture.FT_linear_mipmap_linear)
            tex.setMagfilter(Texture.FT_linear)
            tex.setWrapU(Texture.WM_repeat)
            tex.setWrapV(Texture.WM_repeat)
            if srgb:
                tex.setFormat(Texture.F_srgb)
            return tex
        except Exception as exc:
            log.warning("could not load texture %s: %s", path, exc)
    # 1×1 solid fallback
    tex.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_rgb)
    tex.setRamImage(fallback_rgb)
    return tex


@dataclass
class TerrainData:
    vertices:     np.ndarray   # (N, 3) float32  — (east, north, height)
    normals:      np.ndarray   # (N, 3) float32
    indices:      np.ndarray   # (T*3,) uint32
    uvs:          np.ndarray   # (N, 2) float32
    heightmap:    np.ndarray   # (G, G) float32 — zero-centred at origin
    radius_m:     float
    origin_alt_m: float        # absolute altitude of the scene origin


class Terrain(MapEntity):
    """Terrain mesh sampled from Terrarium tiles.

    Build order::

        sea = Sea(osm, frame, radius_m)
        sea.build()                         # extracts sea polygon
        terrain = Terrain(..., sea_polygon=sea.polygon)
        terrain.build()
        sea.finalize(terrain)               # sea needs terrain altitude

    After ``build()``, ``terrain.data`` is available for dependent entities
    (Sea, Water).
    """

    SHADER = "terrain"

    def __init__(self, frame: Frame, radius_m: float, grid: int,
                 hgt_loader: Callable, sea_polygon=None,
                 tex_paths: dict | None = None) -> None:
        self._frame = frame
        self._radius_m = radius_m
        self._grid = grid
        self._hgt_loader = hgt_loader
        self._sea_polygon = sea_polygon
        self._tex_paths = tex_paths or {}
        self._data: TerrainData | None = None

    def build(self) -> None:
        frame, radius_m, grid = self._frame, self._radius_m, self._grid
        step_m = 2.0 * radius_m / (grid - 1)
        bbox = frame.bbox_ll(radius_m, pad_m=step_m)
        mosaic = self._hgt_loader(bbox)

        eastings  = np.linspace(-radius_m, radius_m, grid)
        northings = np.linspace( radius_m, -radius_m, grid)
        east_g, north_g = np.meshgrid(eastings, northings)
        lon_g, lat_g = frame.to_ll(east_g, north_g)
        heights = mosaic.sample(lat_g, lon_g).astype(np.float32)

        h0 = float(mosaic.sample(np.array([frame.lat0]), np.array([frame.lon0]))[0])
        heightmap = heights - h0

        if self._sea_polygon is not None and not self._sea_polygon.is_empty:
            import shapely
            inside_sea = shapely.contains_xy(
                self._sea_polygon, east_g.ravel(), north_g.ravel(),
            ).reshape(grid, grid)
            sea_clamp = float(-h0 - 0.8)
            heightmap = heightmap.copy()
            heightmap[inside_sea] = np.minimum(heightmap[inside_sea], sea_clamp)
            log.info("sea clamp: pushed %d/%d vertices to %.1f m",
                     int(inside_sea.sum()), inside_sea.size, sea_clamp)

        x = east_g.astype(np.float32).ravel()
        y = north_g.astype(np.float32).ravel()
        vertices = np.stack([x, y, heightmap.ravel()], axis=-1)

        gy_row, gx_col = np.gradient(heightmap, step_m, step_m)
        norms = np.stack([-gx_col, gy_row, np.ones_like(gx_col)],
                         axis=-1).astype(np.float32).reshape(-1, 3)
        norms /= np.linalg.norm(norms, axis=1, keepdims=True).clip(min=1e-8)

        uvs = np.stack([vertices[:, 0], vertices[:, 1]], axis=-1).astype(np.float32)

        g = grid
        r = np.arange(g - 1, dtype=np.uint32)[:, None]
        c = np.arange(g - 1, dtype=np.uint32)[None, :]
        i00 = r * g + c;  i10 = (r+1)*g + c
        i11 = (r+1)*g + (c+1);  i01 = r*g + (c+1)
        indices = np.concatenate([
            np.stack([i10, i11, i01], axis=-1).reshape(-1, 3),
            np.stack([i10, i01, i00], axis=-1).reshape(-1, 3),
        ]).ravel()

        self._data = TerrainData(vertices=vertices, normals=norms, indices=indices,
                                 uvs=uvs, heightmap=heightmap, radius_m=radius_m,
                                 origin_alt_m=h0)
        log.info("terrain: %d verts, %d tris, h=[%.1f..%.1f]",
                 len(vertices), len(indices) // 3,
                 float(heightmap.min()), float(heightmap.max()))

    @property
    def data(self) -> TerrainData:
        if self._data is None:
            raise RuntimeError("Terrain.build() has not been called")
        return self._data

    def attach_to(self, parent) -> None:
        from panda3d.core import Filename, Texture
        from osm3denv.render.helpers import attach_mesh, load_shader
        td = self.data
        np_ = attach_mesh(parent, "terrain", td.vertices, td.normals, td.uvs, td.indices)
        shader = load_shader(self.SHADER)
        if shader:
            np_.setShader(shader)
            np_.setShaderInput("u_origin_alt_m", float(td.origin_alt_m))
            np_.setShaderInput("u_radius_m",     float(td.radius_m))
            np_.setShaderInput("u_tex_scale",     20.0)
            np_.setShaderInput("u_bump_strength", 3.0)
            # Blank 1×1 splatmap — Roads.attach_to() overrides this if roads exist.
            for smap_name in ("u_road_splatmap", "u_beach_splatmap"):
                blank = Texture(smap_name)
                blank.setup2dTexture(1, 1, Texture.T_unsigned_byte, Texture.F_luminance)
                blank.setRamImage(bytes([0]))
                np_.setShaderInput(smap_name, blank)
            # Terrain textures (sand / grass / rock)
            _FALLBACKS = {
                "sand":       (bytes([199, 184, 133]), bytes([128, 128, 255])),
                "grass":      (bytes([ 77, 125,  41]), bytes([128, 128, 255])),
                "rock":       (bytes([112, 102,  89]), bytes([128, 128, 255])),
                "beach_sand": (bytes([210, 195, 155]), bytes([128, 128, 255])),
            }
            for name, (fb_col, fb_nrm) in _FALLBACKS.items():
                col_path = self._tex_paths.get(name, {}).get("color")
                nrm_path = self._tex_paths.get(name, {}).get("normal")
                col_tex = _load_texture(col_path, fb_col, srgb=True)
                nrm_tex = _load_texture(nrm_path, fb_nrm, srgb=False)
                np_.setShaderInput(f"u_{name}_col", col_tex)
                np_.setShaderInput(f"u_{name}_nrm", nrm_tex)
