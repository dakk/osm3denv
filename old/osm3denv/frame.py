"""Local ENU (east-north-up) projection around an input lat/lon.

Uses azimuthal-equidistant centered on the origin so distances within a few km
are accurate to centimeters. All returned coordinates are in meters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class Frame:
    lat0: float
    lon0: float
    _fwd: Transformer
    _inv: Transformer

    def to_enu(self, lon, lat):
        """Project (lon, lat) arrays (degrees) to (east, north) arrays (meters)."""
        return self._fwd.transform(lon, lat)

    def to_ll(self, east, north):
        """Inverse project (east, north) arrays (meters) to (lon, lat) arrays (degrees)."""
        return self._inv.transform(east, north)

    def bbox_ll(self, radius_m: float, pad_m: float = 0.0) -> tuple[float, float, float, float]:
        """Return (min_lon, min_lat, max_lon, max_lat) enclosing a square of side 2*(radius+pad)."""
        r = radius_m + pad_m
        # Sample 4 corners of the square and take their geographic extent.
        e = np.array([-r, r, r, -r])
        n = np.array([-r, -r, r, r])
        lon, lat = self.to_ll(e, n)
        return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def make_frame(lat0: float, lon0: float) -> Frame:
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
    )
    fwd = Transformer.from_crs(4326, local, always_xy=True)
    inv = Transformer.from_crs(local, 4326, always_xy=True)
    return Frame(lat0=lat0, lon0=lon0, _fwd=fwd, _inv=inv)
