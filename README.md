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


## Licence

Released under the [MIT License](LICENSE).
