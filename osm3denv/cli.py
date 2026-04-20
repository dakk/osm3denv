from __future__ import annotations

import logging
from pathlib import Path

import click

from osm3denv import logging as _logging
from osm3denv.config import Config, default_cache_dir
from osm3denv.frame import make_frame

log = logging.getLogger("osm3denv")


@click.command()
@click.option("--lat", type=float, required=True, help="Latitude in degrees (WGS84).")
@click.option("--lon", type=float, required=True, help="Longitude in degrees (WGS84).")
@click.option("--radius", "radius_m", type=float, default=1500.0, show_default=True,
              help="Half-extent of the square scene in meters (capped at 5000).")
@click.option("--grid", type=int, default=201, show_default=True,
              help="Terrain grid vertices per side.")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None,
              help="Cache directory (default: platformdirs user cache).")
@click.option("--no-buildings", is_flag=True, help="Skip OSM buildings.")
@click.option("--no-roads", is_flag=True, help="Skip OSM roads/railways.")
@click.option("--no-water", is_flag=True, help="Skip OSM water features.")
@click.option("--fetch-only", is_flag=True,
              help="Download data and build caches, then exit without rendering.")
@click.option("--refresh-cache", is_flag=True,
              help="Ignore cached SRTM/OSM data for this run and re-download.")
@click.option("--classmap-size", type=int, default=2048, show_default=True,
              help="Pixel resolution of the landuse class-map (square).")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity (-v, -vv).")
def main(lat, lon, radius_m, grid, cache_dir, no_buildings, no_roads, no_water,
         fetch_only, refresh_cache, classmap_size, verbose):
    """Generate a 3D scene around (lat, lon) from OSM + SRTM data."""
    _logging.configure(verbose)

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise click.BadParameter("lat must be in [-90,90], lon in [-180,180].")
    radius_m = min(max(radius_m, 100.0), 5000.0)

    layers = {"terrain"}
    if not no_buildings:
        layers.add("buildings")
    if not no_roads:
        layers.add("roads")
    if not no_water:
        layers.add("water")

    cfg = Config(
        lat=lat, lon=lon, radius_m=radius_m, grid=grid,
        cache_dir=cache_dir or default_cache_dir(),
        fetch_only=fetch_only,
        refresh_cache=refresh_cache,
        classmap_size=max(256, min(8192, int(classmap_size))),
        layers=frozenset(layers),
    )
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("origin=(%.6f,%.6f) radius=%.0fm cache=%s layers=%s",
             cfg.lat, cfg.lon, cfg.radius_m, cfg.cache_dir, sorted(cfg.layers))

    frame = make_frame(cfg.lat, cfg.lon)
    run(cfg, frame)


def run(cfg: Config, frame) -> None:
    from osm3denv.fetch import srtm as srtm_fetch
    from osm3denv.mesh import terrain as terrain_mesh

    # OSM first so the sea mask (if any) can be applied to the terrain before meshing.
    # Areas + trees are always on, so we always need OSM data.
    from osm3denv.fetch import osm as osm_fetch
    osm_data = osm_fetch.fetch(frame=frame, radius_m=cfg.radius_m,
                               cache_dir=cfg.osm_cache,
                               refresh=cfg.refresh_cache)
    sea_poly = None
    if "water" in cfg.layers:
        from osm3denv.mesh.water import build_sea_polygon
        sea_poly = build_sea_polygon(osm_data, frame, cfg.radius_m)
        if sea_poly is not None:
            log.info("sea polygon area=%.0f m² (%s)", sea_poly.area, sea_poly.geom_type)

    terrain_data = terrain_mesh.build(
        frame=frame, radius_m=cfg.radius_m, grid=cfg.grid,
        hgt_loader=srtm_fetch.loader(cfg.srtm_cache, refresh=cfg.refresh_cache),
        sea_mask=sea_poly,
    )
    log.info("terrain: %d verts, %d tris, h=[%.1f..%.1f]",
             len(terrain_data.vertices), len(terrain_data.indices) // 3,
             float(terrain_data.heightmap.min()), float(terrain_data.heightmap.max()))

    buildings_data = roads_data = water_data = None
    trees_data = None
    area_meshes: list = []
    if osm_data is not None:
        from osm3denv.mesh import areas as amesh
        from osm3denv.mesh import trees as tmesh

        area_meshes = amesh.build(osm_data, frame, terrain_data.sampler)
        log.info("areas: %d materials, %d polygons",
                 len(area_meshes), sum(m.count for m in area_meshes))
        trees_data = tmesh.build(osm_data, frame, terrain_data.sampler,
                                  radius_m=cfg.radius_m)
        log.info("trees: %d", trees_data.count)

        if "buildings" in cfg.layers:
            from osm3denv.mesh import buildings as bmesh
            buildings_data = bmesh.build(osm_data, frame, terrain_data.sampler)
            log.info("buildings: %d", buildings_data.count)
        if "roads" in cfg.layers:
            from osm3denv.mesh import roads as rmesh
            roads_data = rmesh.build(osm_data, frame, terrain_data.sampler)
            log.info("roads: %d", roads_data.count)
        if "water" in cfg.layers:
            from osm3denv.mesh import water as wmesh
            water_data = wmesh.build(
                osm_data, frame, terrain_data.sampler,
                radius_m=cfg.radius_m,
                sea_y=-terrain_data.origin_alt_m,
                sea_polygon=sea_poly,
            )
            log.info("water: %d", water_data.count)

    if cfg.fetch_only:
        log.info("fetch-only: done.")
        return

    from osm3denv.render.app import run_viewer
    run_viewer(terrain=terrain_data, buildings=buildings_data,
               roads=roads_data, water=water_data,
               areas=area_meshes, trees=trees_data)


if __name__ == "__main__":
    main()
