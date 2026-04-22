from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir


@dataclass(frozen=True)
class Config:
    lat: float
    lon: float
    radius_m: float
    grid: int
    cache_dir: Path
    fetch_only: bool
    refresh_cache: bool
    dem_zoom: int | None = None   # None = auto-select per terrain step size

    @property
    def srtm_cache(self) -> Path:
        p = self.cache_dir / "srtm"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def osm_cache(self) -> Path:
        p = self.cache_dir / "osm"
        p.mkdir(parents=True, exist_ok=True)
        return p


def default_cache_dir() -> Path:
    return Path(user_cache_dir("osm3denv"))
