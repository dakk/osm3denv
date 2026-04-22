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
@click.option("-v", "--verbose", count=True, help="Increase log verbosity (-v, -vv).")
def main(lat, lon, radius_m, grid, cache_dir, fetch_only, refresh_cache, verbose):
    """Render a 3D terrain around (lat, lon) from SRTM and cache OSM data."""
    _logging.configure(verbose)

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise click.BadParameter("lat must be in [-90,90], lon in [-180,180].")
    radius_m = min(max(radius_m, 100.0), 5000.0)

    cfg = Config(
        lat=lat, lon=lon, radius_m=radius_m, grid=grid,
        cache_dir=cache_dir or default_cache_dir(),
        fetch_only=fetch_only,
        refresh_cache=refresh_cache,
    )
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("origin=(%.6f,%.6f) radius=%.0fm grid=%d cache=%s",
             cfg.lat, cfg.lon, cfg.radius_m, cfg.grid, cfg.cache_dir)

    frame = make_frame(cfg.lat, cfg.lon)
    run(cfg, frame)


def run(cfg: Config, frame) -> None:
    from osm3denv.fetch import osm as osm_fetch
    from osm3denv.fetch import srtm as srtm_fetch
    from osm3denv.mesh import coastline as coastline_mesh
    from osm3denv.mesh import sea as sea_mesh
    from osm3denv.mesh import terrain as terrain_mesh

    osm_data = osm_fetch.fetch(frame=frame, radius_m=cfg.radius_m,
                               cache_dir=cfg.osm_cache,
                               refresh=cfg.refresh_cache)
    log.info("osm: %d ways, %d relations, %d nodes",
             len(osm_data.ways), len(osm_data.relations), len(osm_data.nodes))

    sea_polygon = sea_mesh.build_sea_polygon(osm_data, frame, cfg.radius_m)
    if sea_polygon is not None:
        log.info("sea polygon: area=%.0f m² (%s)",
                 sea_polygon.area, sea_polygon.geom_type)
    else:
        log.info("sea polygon: none (no OSM coastlines in bbox — "
                 "run with --refresh-cache if you expected some)")

    terrain = terrain_mesh.build(
        frame=frame, radius_m=cfg.radius_m, grid=cfg.grid,
        hgt_loader=srtm_fetch.loader(cfg.srtm_cache, refresh=cfg.refresh_cache),
        sea_polygon=sea_polygon,
    )
    log.info("terrain: %d verts, %d tris, h=[%.1f..%.1f]",
             len(terrain.vertices), len(terrain.indices) // 3,
             float(terrain.heightmap.min()), float(terrain.heightmap.max()))

    coastline = coastline_mesh.build(osm_data, frame, cfg.radius_m)
    log.info("coastline: %d polylines (%d vertices)",
             len(coastline.polylines),
             sum(len(p) for p in coastline.polylines))

    if cfg.fetch_only:
        log.info("fetch-only: done.")
        return

    # Sea sits 0.3 m below absolute sea level so low shore terrain (SRTM
    # cells near 0 m) does not Z-fight the plane at the coast.
    sea_z = -terrain.origin_alt_m - 0.3
    from osm3denv.render.app import run_viewer
    run_viewer(terrain, coastline=coastline, sea_z=sea_z, sea_polygon=sea_polygon)


if __name__ == "__main__":
    main()
