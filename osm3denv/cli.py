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
@click.option("--fetch-only", is_flag=True,
              help="Download data and build caches, then exit without rendering.")
@click.option("--refresh-cache", is_flag=True,
              help="Ignore cached SRTM/OSM data for this run and re-download.")
@click.option("--dem-zoom", type=click.IntRange(10, 15), default=None, show_default=True,
              help="Terrarium tile zoom level (10-15). Auto-selected if omitted; "
                   "use 15 for ~4.5 m/px, 14 for ~9 m/px.")
@click.option("--no-roads",      is_flag=True, help="Skip road rendering.")
@click.option("--no-powerlines", is_flag=True, help="Skip power line rendering.")
@click.option("--no-vegetation", is_flag=True, help="Skip vegetation rendering.")
@click.option("--no-buildings",  is_flag=True, help="Skip building rendering.")
@click.option("-v", "--verbose", count=True, help="Increase log verbosity (-v, -vv).")
def main(lat, lon, radius_m, grid, cache_dir, fetch_only, refresh_cache, dem_zoom,
         no_roads, no_powerlines, no_vegetation, no_buildings, verbose):
    """Render a 3D terrain around (lat, lon) from SRTM and OSM data."""
    _logging.configure(verbose)

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise click.BadParameter("lat must be in [-90,90], lon in [-180,180].")
    radius_m = round(radius_m / 1000.0) * 1000

    cfg = Config(
        lat=lat, lon=lon, radius_m=radius_m, grid=grid,
        cache_dir=cache_dir or default_cache_dir(),
        fetch_only=fetch_only,
        refresh_cache=refresh_cache,
        dem_zoom=dem_zoom,
    )
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("origin=(%.6f,%.6f) radius=%.0fm grid=%d cache=%s",
             cfg.lat, cfg.lon, cfg.radius_m, cfg.grid, cfg.cache_dir)

    frame = make_frame(cfg.lat, cfg.lon)
    run(cfg, frame, no_roads=no_roads, no_powerlines=no_powerlines,
        no_vegetation=no_vegetation, no_buildings=no_buildings)


def run(cfg: Config, frame, *,
        no_roads: bool = False,
        no_powerlines: bool = False,
        no_vegetation: bool = False,
        no_buildings: bool = False) -> None:
    from osm3denv.entities.beach import Beach
    from osm3denv.entities.buildings import Buildings
    from osm3denv.entities.clouds import Clouds
    from osm3denv.entities.coastline import Coastline
    from osm3denv.entities.powerlines import PowerLines
    from osm3denv.entities.roads import Roads
    from osm3denv.entities.sea import Sea
    from osm3denv.entities.terrain import Terrain
    from osm3denv.entities.vegetation import Vegetation
    from osm3denv.entities.water import Water
    from osm3denv.fetch import osm as osm_fetch
    from osm3denv.fetch import terrarium as dem_fetch
    from osm3denv.fetch import textures as tex_fetch

    osm_data = osm_fetch.fetch(frame=frame, radius_m=cfg.radius_m,
                               cache_dir=cfg.osm_cache, refresh=cfg.refresh_cache)
    log.info("osm: %d ways, %d relations, %d nodes",
             len(osm_data.ways), len(osm_data.relations), len(osm_data.nodes))

    tex_paths      = tex_fetch.fetch(cfg.tex_cache)
    bld_tex_paths  = tex_fetch.fetch_building(cfg.tex_cache)

    from osm3denv.render.minimap import Minimap
    minimap = Minimap(cfg.lat, cfg.lon, cfg.radius_m,
                      cache_dir=cfg.cache_dir / "minimap")
    minimap.fetch()

    # Phase 1 — sea polygon (Terrain needs it to clamp underwater vertices).
    sea = Sea(osm_data, frame, cfg.radius_m)
    sea.build()

    # Phase 2 — terrain mesh.
    terrain = Terrain(
        frame=frame, radius_m=cfg.radius_m, grid=cfg.grid,
        hgt_loader=dem_fetch.loader(cfg.srtm_cache, zoom=cfg.dem_zoom,
                                    refresh=cfg.refresh_cache),
        sea_polygon=sea.polygon,
        tex_paths=tex_paths,
    )
    terrain.build()

    # Phase 3 — everything that depends on terrain data.
    sea.finalize(terrain)

    coastline = Coastline(osm_data, frame, cfg.radius_m, sea_z=sea.sea_z)
    coastline.build()

    water = Water(osm_data, frame, cfg.radius_m, terrain)
    water.build()

    beach = Beach(osm_data, frame, cfg.radius_m)
    beach.build()

    clouds = Clouds(cfg.radius_m)
    clouds.build()

    entities = [terrain, sea, coastline, water, beach, clouds]

    if not no_roads:
        roads = Roads(osm_data, frame, cfg.radius_m)
        roads.build()
        entities.append(roads)

    if not no_powerlines:
        powerlines = PowerLines(osm_data, frame, cfg.radius_m, terrain)
        powerlines.build()
        entities.append(powerlines)

    if not no_vegetation:
        vegetation = Vegetation(osm_data, frame, cfg.radius_m, terrain)
        vegetation.build()
        entities.append(vegetation)

    if not no_buildings:
        buildings = Buildings(osm_data, frame, cfg.radius_m, terrain, bld_tex_paths)
        buildings.build()
        entities.append(buildings)

    if cfg.fetch_only:
        log.info("fetch-only: done.")
        return

    from osm3denv.render.app import run_viewer
    run_viewer(terrain, entities=entities, frame=frame, minimap=minimap)


if __name__ == "__main__":
    main()
