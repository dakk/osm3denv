"""OSM feature download via Overpass API.

Returns a lightweight dataclass tree (ways + relations with inline geometry).
Consumers in :mod:`osm3denv.mesh` filter by tags. Raw Overpass JSON is cached
under ``cache_dir`` keyed by bbox; default TTL is 7 days.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from osm3denv.cache import bbox_key, read_json, write_json
from osm3denv.frame import Frame

log = logging.getLogger(__name__)

DEFAULT_OVERPASS = "https://overpass-api.de/api/interpreter"
FALLBACK_OVERPASS = "https://overpass.kumi.systems/api/interpreter"
USER_AGENT = "osm3denv/0.1 (+https://github.com/local/osm3denv)"


@dataclass
class OSMWay:
    id: int
    tags: dict[str, str]
    geometry: list[tuple[float, float]]  # (lon, lat)


@dataclass
class OSMRelation:
    id: int
    tags: dict[str, str]
    rings: list[tuple[str, list[tuple[float, float]]]]  # [(role, [(lon, lat), ...]), ...]


@dataclass
class OSMNode:
    id: int
    tags: dict[str, str]
    lon: float
    lat: float


@dataclass
class OSMData:
    ways: list[OSMWay] = field(default_factory=list)
    relations: list[OSMRelation] = field(default_factory=list)
    nodes: list[OSMNode] = field(default_factory=list)

    def filter_ways(self, predicate) -> list[OSMWay]:
        return [w for w in self.ways if predicate(w.tags)]

    def filter_relations(self, predicate) -> list[OSMRelation]:
        return [r for r in self.relations if predicate(r.tags)]

    def filter_nodes(self, predicate) -> list[OSMNode]:
        return [n for n in self.nodes if predicate(n.tags)]


def _build_query(bbox_ll: tuple[float, float, float, float]) -> str:
    min_lon, min_lat, max_lon, max_lat = bbox_ll
    bb = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    return f"""[out:json][timeout:90];
(
  way["building"]({bb});
  way["building:part"]({bb});
  relation["building"]({bb});
  way["highway"]({bb});
  way["railway"]({bb});
  way["natural"="water"]({bb});
  relation["natural"="water"]({bb});
  way["waterway"]({bb});
  way["natural"="coastline"]({bb});
  way["landuse"]({bb});
  relation["landuse"]({bb});
  way["leisure"]({bb});
  relation["leisure"]({bb});
  way["place"="square"]({bb});
  relation["place"="square"]({bb});
  way["highway"="pedestrian"]["area"="yes"]({bb});
  way["amenity"="marketplace"]({bb});
  relation["amenity"="marketplace"]({bb});
  way["natural"~"^(wood|scrub|grassland|heath|bare_rock|sand|beach|scree)$"]({bb});
  relation["natural"~"^(wood|scrub|grassland|heath)$"]({bb});
  node["natural"="tree"]({bb});
  node["highway"="street_lamp"]({bb});
  node["amenity"="bench"]({bb});
  way["amenity"="fountain"]({bb});
  relation["amenity"="fountain"]({bb});
  node["amenity"="fountain"]({bb});
  way["man_made"~"^(obelisk|column)$"]({bb});
  node["man_made"~"^(obelisk|column)$"]({bb});
  node["highway"="crossing"]({bb});
);
out geom;
"""


def _fetch_overpass(query: str) -> dict:
    last_err: Exception | None = None
    headers = {"User-Agent": USER_AGENT,
               "Accept": "application/json",
               "Accept-Encoding": "gzip, deflate"}
    for url in (DEFAULT_OVERPASS, FALLBACK_OVERPASS):
        for attempt in range(3):
            try:
                log.info("overpass POST %s (attempt %d)", url, attempt + 1)
                r = requests.post(url, data={"data": query},
                                  headers=headers, timeout=120)
                if r.status_code in (429, 502, 503, 504):
                    wait = 10 * (attempt + 1)
                    log.warning("status %d (server busy), backing off %ds",
                                r.status_code, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as exc:
                last_err = exc
                log.warning("overpass attempt failed: %s", exc)
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Overpass request failed on all endpoints: {last_err}")


def _parse(data: dict) -> OSMData:
    out = OSMData()
    for el in data.get("elements", []):
        tp = el.get("type")
        tags = el.get("tags", {}) or {}
        if tp == "way":
            geom = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
            if len(geom) >= 2:
                out.ways.append(OSMWay(id=el["id"], tags=tags, geometry=geom))
        elif tp == "relation":
            rings = []
            for m in el.get("members", []):
                if m.get("type") != "way" or "geometry" not in m:
                    continue
                role = m.get("role") or "outer"
                ring = [(g["lon"], g["lat"]) for g in m["geometry"]]
                if len(ring) >= 2:
                    rings.append((role, ring))
            if rings:
                out.relations.append(OSMRelation(id=el["id"], tags=tags, rings=rings))
        elif tp == "node" and tags:
            out.nodes.append(OSMNode(id=el["id"], tags=tags,
                                     lon=el["lon"], lat=el["lat"]))
        # Tag-less nodes are geometry-only refs for ways; skipped.
    return out


def fetch(*, frame: Frame, radius_m: float, cache_dir: Path,
          ttl_days: int = 7, refresh: bool = False) -> OSMData:
    bbox = frame.bbox_ll(radius_m)
    key = bbox_key(bbox)
    path = cache_dir / f"{key}.json"
    if not refresh:
        cached = read_json(path, max_age_s=ttl_days * 86400)
        if cached is not None:
            log.info("osm cache hit: %s (%d elements)",
                     path.name, len(cached.get("elements", [])))
            return _parse(cached)
    q = _build_query(bbox)
    data = _fetch_overpass(q)
    write_json(path, data)
    log.info("osm cached %d elements -> %s",
             len(data.get("elements", [])), path)
    return _parse(data)
