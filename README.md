# osm3denv

`osm3denv` is a Python command-line tool that turns any geographic coordinate into
a real-time, interactive 3D scene you can fly/walk through. Given a lat/lon, it:

1. Downloads the surrounding elevation data (SRTM 1 arc-second) and
   OpenStreetMap features (buildings, roads, water, landuse, …).
2. Meshes everything into a local East-North-Up frame centred on the point.
3. Opens an [Ogre3D](https://ogrecave.github.io/ogre/) window with the scene,
   entirely rendered by custom GLSL shaders — no texture files, everything
   procedurally generated.

The scene is not a game, but the foundation for one: walk a real neighbourhood,
inspect a coastline, preview a building footprint in context, or use it as a
starting terrain for a sim.

## What you see

Given a coordinate (say `41.9141, 12.4823` — Rome, Villa Borghese), the
viewer renders:

| Layer | Source | Geometry | Material |
|---|---|---|---|
| Terrain | SRTM 1-arc-second heightmap | Regular ENU grid (201×201 by default) | `terrain.frag` — grass / rock / sand blended by slope + altitude |
| Buildings | OSM `building`, `building:part` | Extruded footprints with roofs and walls | `buildings.frag` — brick walls with mortar grid, procedural windows, terracotta tile roofs |
| Roads & railways | OSM `highway`, `railway` | Polyline buffered to width, draped on terrain | `asphalt.frag` — noise grain + pebble sparkle |
| Bridges / tunnels | `bridge=yes` / `tunnel=yes` | Bridges lifted to a flat y above terrain; tunnels hidden | same road shader |
| Water areas & rivers | OSM `natural=water`, `waterway=*` | Flat polygons + buffered ribbons | `water.frag` — procedural ripple normals with specular highlight |
| Sea | OSM `natural=coastline` | Filled halfspaces derived from coastline direction (OSM right-hand rule) | water shader |
| Landuse / vegetation | OSM `landuse`, `leisure`, `natural` (woods, sand, rock) | Draped polygons following terrain surface | `area.frag` — one shader switching on `area_class` to produce vegetation / residential / commercial / industrial / farmland / sand / rock |
| Trees | OSM `natural=tree` nodes | Low-poly 6-sided cones with per-tree height/orientation | flat-colour material |

Sea level is inferred from the origin's absolute SRTM altitude: the sea plane
sits at `y = -origin_alt_m`, so a coastal scene has sea right where it
belongs, and an inland scene never shows a spurious ocean.

## Quick start

```bash
# Requires Python 3.12 (see note below) and a working OpenGL driver.
python3.12 -m venv --copies venv
./venv/bin/pip install -e .

# Interactive viewer around the Colosseum
./venv/bin/python -m osm3denv --lat 41.8902 --lon 12.4922 --radius 1500 -v
```

Controls:

- **Mouse** — free look (cursor is grabbed)
- **W A S D** — walk
- **Space / E** — up, **Ctrl / Q** — down
- **Shift** — 5× sprint
- **Esc** — quit

First run on a given bbox downloads ~25 MB of SRTM tile + one Overpass query;
subsequent runs on the same coord are near-instant thanks to on-disk caching
(`~/.cache/osm3denv/` by default; override with `--cache-dir`).

### Other useful flags

```bash
--radius 2000          # scene half-extent in metres (capped at 5000)
--grid 301             # terrain grid vertices per side
--no-buildings         # skip building extrusion
--no-roads             # skip highways / railways
--no-water             # skip water features + sea
--fetch-only           # build caches and exit without rendering (CI-friendly)
--refresh-cache        # ignore cached SRTM/OSM for this run
-v / -vv               # verbose logging
```

### Some suggested coordinates

```
Rome, Colosseum         --lat 41.8902  --lon 12.4922  --radius 1000
Rome, Villa Borghese    --lat 41.9141  --lon 12.4823  --radius 1500
Central Park NYC        --lat 40.7829  --lon -73.9654 --radius 2000
Innsbruck (Alps)        --lat 47.2692  --lon 11.4041  --radius 2000
Sydney Opera House      --lat -33.8568 --lon 151.2153 --radius 1500
Cagliari waterfront     --lat 39.2182  --lon 9.1167   --radius 4000
```

## How it works

```
 cli.py
   │
   ├──> fetch/srtm.py    ── tilezen S3 `.hgt.gz`  ── → HGT arrays (cached)
   │
   ├──> fetch/osm.py     ── Overpass API           ── → ways / relations / nodes (JSON, cached)
   │
   ├──> frame.py         ── pyproj local azimuthal-equidistant projection
   │
   ├──> mesh/
   │     ├── terrain.py    SRTM → ENU grid mesh (triangle-plane TerrainSampler)
   │     ├── buildings.py  polygons → extruded solids (earcut caps + ring walls)
   │     ├── roads.py      polylines → buffered ribbons, bridge lift, tunnel skip
   │     ├── water.py      lakes, rivers, sea from coastline directions
   │     ├── areas.py      landuse/leisure/natural → draped polygons per class
   │     ├── trees.py      natural=tree nodes → instanced 6-sided cones
   │     ├── drape.py      polygon → densified ribbon at exact terrain y
   │     └── geom.py       shared OSM → Shapely polygon helpers
   │
   └──> render/
         ├── app.py        OgreBites application, scene graph, camera, resource loader
         ├── upload.py     numpy arrays → Ogre ManualObject
         ├── materials.py  idempotent material factories (scripts supply the real defs)
         └── camera.py     free-look InputListener (WASD, mouse-grab, sprint)

assets/
  shaders/*.{vert,frag}      GLSL 150 custom shaders
  materials/osm3d.{program,material}  Ogre script definitions
```

### Coordinate frame

All geometry lives in a local ENU frame centered on the input coordinate,
produced via a `+proj=aeqd` pyproj transformer. Within a few kilometres the
frame is accurate to centimetres; it also avoids the float-precision problems
you'd hit working directly in global web-mercator metres.

The Ogre mapping is `(x = east, y = height, z = -north)`, right-handed with
`+y` up.

### Shader-only materials

All colours are generated inside custom GLSL fragment shaders using hash noise
+ FBM + Worley cells — no PNG assets are shipped. The base-material set lives
in [`assets/shaders/`](osm3denv/assets/shaders/):

- `terrain.frag` — grass × rock × sand blended by slope and altitude
- `buildings.frag` — brick rows with mortar, grid of alpha-free "windows",
  terracotta tile roof
- `asphalt.frag` — fine grain + occasional bright pebble inclusions
- `water.frag` — analytic ripple normals derived from four directional sines,
  Blinn-Phong specular, semi-transparent
- `area.frag` — switch statement on `area_class` uniform for vegetation,
  residential, commercial, industrial, farmland, sand, rock

Materials are declared in [`assets/materials/osm3d.material`](osm3denv/assets/materials/osm3d.material)
and the shader programs in [`osm3d.program`](osm3denv/assets/materials/osm3d.program).
Ogre's RTShader System is still in the pipeline as a fallback for any future
fixed-function materials.

### Caching

- **SRTM** tiles are cached forever as uncompressed `.hgt` files.
- **Overpass** responses are cached as raw JSON keyed by bbox + date, 7-day TTL.

The cache is a plain directory — safe to inspect or `rm -rf`.

## Dependencies

Runtime:

- Python **3.12** (pinned because the `ogre-python` PyPI wheels currently
  ship for cp3.10 / cp3.12 / cp3.14)
- [`ogre-python`](https://pypi.org/project/ogre-python/) — official SWIG
  bindings for Ogre 14.x
- `pyproj`, `shapely>=2`, `numpy`, `mapbox-earcut`, `rasterio`, `overpy`,
  `requests`, `click`, `platformdirs`

System requirements:

- A working OpenGL 3.3+ driver (GL3+ is what Ogre 14 uses by default)
- For headless runs, `Xvfb :99 -screen 0 1920x1080x24 &` + `export DISPLAY=:99`
  is sufficient — or use `--fetch-only` to exercise the pipeline without a
  window.

## Limitations

- Terrain grid is uniform; no LOD, no streaming — fine up to ~5 km radius.
- Building heights come from OSM `height` / `building:levels` tags; missing
  tags fall back to 6 m (≈ 2 floors).
- Sea rendering assumes true sea level is `y = 0` in absolute SRTM altitude.
  Inland lakes are rendered separately via `natural=water`.
- Bridges render as flat lifted ribbons without pier geometry.
- Overpass has per-IP rate limits. If you re-run many bboxes in a minute you
  may get 429 / 504 — the client retries with exponential backoff, but
  patience helps.

## Licence

No licence has been declared yet. Assume "all rights reserved" until a
`LICENSE` file is added.
