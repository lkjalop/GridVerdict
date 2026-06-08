"""OpenMeteo site-specific solar irradiance + wind speed client.

Fetches real-time and forecast weather variables at AEMO generator lat/lon.
Used to provide site-level renewable output context beyond BOM regional averages.

Data source: Open-Meteo API (free, no auth required, CC-BY license).
  Endpoint: https://api.open-meteo.com/v1/forecast

Variables fetched per generator site:
  - shortwave_radiation (W/m²) — global horizontal irradiance for solar farms
  - wind_speed_10m (km/h) — wind speed at hub height proxy for wind farms
  - temperature_2m (°C) — demand driver proxy

NEM generator coordinates are hardcoded from AEMO MMS GENCONDATA / publicly
published NEM registration tables. Add new sites by extending _GENERATOR_SITES.

This client is fault-tolerant: any failure returns an empty dict so the rest
of the scatter-gather pipeline is unaffected.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(15.0)
_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Prominent NEM generator sites with lat/lon
# Source: AEMO MMS GENCONDATA registration table (public)
_GENERATOR_SITES: list[dict[str, Any]] = [
    # --- Solar farms ---
    {"duid": "HORNSD01",  "name": "Hornsdale Solar",       "region": "SA1",  "lat": -33.057, "lon": 138.665, "type": "solar"},
    {"duid": "DAYDSF1",   "name": "Daydream Solar",        "region": "QLD1", "lat": -20.642, "lon": 147.315, "type": "solar"},
    {"duid": "WALGRVS1",  "name": "Walgrove Solar",        "region": "NSW1", "lat": -33.844, "lon": 150.880, "type": "solar"},
    {"duid": "BROKENH1",  "name": "Broken Hill Solar",     "region": "NSW1", "lat": -31.962, "lon": 141.449, "type": "solar"},
    {"duid": "NEREGLS1",  "name": "Neregula Solar",        "region": "NSW1", "lat": -33.558, "lon": 148.612, "type": "solar"},
    {"duid": "MULGBWF1",  "name": "Mulgalong Solar",       "region": "NSW1", "lat": -33.840, "lon": 147.600, "type": "solar"},
    # --- Wind farms ---
    {"duid": "HPRG1",     "name": "Hornsdale Wind",        "region": "SA1",  "lat": -33.069, "lon": 138.685, "type": "wind"},
    {"duid": "WGWF1",     "name": "Waterloo Wind",         "region": "SA1",  "lat": -33.936, "lon": 139.024, "type": "wind"},
    {"duid": "ARWF1",     "name": "Ararat Wind",           "region": "VIC1", "lat": -37.314, "lon": 143.014, "type": "wind"},
    {"duid": "MEWF1",     "name": "Mount Mercer Wind",     "region": "VIC1", "lat": -37.819, "lon": 144.095, "type": "wind"},
    {"duid": "BUNGWF1",   "name": "Bungaban Wind",         "region": "QLD1", "lat": -26.978, "lon": 150.890, "type": "wind"},
    {"duid": "CAPTL_WF",  "name": "Capital Wind",          "region": "NSW1", "lat": -35.545, "lon": 149.378, "type": "wind"},
    {"duid": "WOOLNTH1",  "name": "Woolnorth Wind",        "region": "TAS1", "lat": -40.731, "lon": 144.692, "type": "wind"},
]

# Map region → sites
_SITES_BY_REGION: dict[str, list[dict[str, Any]]] = {}
for _s in _GENERATOR_SITES:
    _SITES_BY_REGION.setdefault(_s["region"], []).append(_s)


@dataclass
class SiteWeather:
    duid: str
    name: str
    region: str
    generator_type: str           # "solar" | "wind"
    lat: float
    lon: float
    current_radiation_wm2: float | None  # W/m² — solar only
    current_wind_kmh: float | None       # km/h — wind only
    current_temp_c: float | None
    forecast_radiation: list[float] = field(default_factory=list)   # next 6h, hourly
    forecast_wind:      list[float] = field(default_factory=list)
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "OPEN_METEO"

    def to_dict(self) -> dict[str, Any]:
        return {
            "duid": self.duid,
            "name": self.name,
            "region": self.region,
            "type": self.generator_type,
            "lat": self.lat,
            "lon": self.lon,
            "current_radiation_wm2": self.current_radiation_wm2,
            "current_wind_kmh": self.current_wind_kmh,
            "current_temp_c": self.current_temp_c,
            "forecast_radiation_6h": self.forecast_radiation,
            "forecast_wind_6h": self.forecast_wind,
            "as_of": self.as_of.isoformat(),
            "source": self.source,
        }

    @property
    def low_solar_alert(self) -> bool:
        """True when current radiation is below 100 W/m² during daylight hours (8-18 UTC+10)."""
        if self.current_radiation_wm2 is None:
            return False
        local_hour = (self.as_of.hour + 10) % 24
        return 8 <= local_hour <= 18 and self.current_radiation_wm2 < 100.0

    @property
    def low_wind_alert(self) -> bool:
        """True when wind speed is below cut-in speed (~12 km/h)."""
        return self.current_wind_kmh is not None and self.current_wind_kmh < 12.0


@dataclass
class RegionSiteWeather:
    """Aggregated site-level weather for a NEM region."""
    region: str
    sites: list[SiteWeather] = field(default_factory=list)
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def solar_sites(self) -> list[SiteWeather]:
        return [s for s in self.sites if s.generator_type == "solar"]

    @property
    def wind_sites(self) -> list[SiteWeather]:
        return [s for s in self.sites if s.generator_type == "wind"]

    @property
    def avg_radiation_wm2(self) -> float | None:
        vals = [s.current_radiation_wm2 for s in self.solar_sites if s.current_radiation_wm2 is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    @property
    def avg_wind_kmh(self) -> float | None:
        vals = [s.current_wind_kmh for s in self.wind_sites if s.current_wind_kmh is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "as_of": self.as_of.isoformat(),
            "site_count": len(self.sites),
            "avg_radiation_wm2": self.avg_radiation_wm2,
            "avg_wind_kmh": self.avg_wind_kmh,
            "low_solar_alert": any(s.low_solar_alert for s in self.solar_sites),
            "low_wind_alert": any(s.low_wind_alert for s in self.wind_sites),
            "sites": [s.to_dict() for s in self.sites],
        }


async def _fetch_site_weather(
    client: httpx.AsyncClient,
    site: dict[str, Any],
) -> SiteWeather | None:
    """Fetch current + 6h forecast weather for a single generator site."""
    params = {
        "latitude":   site["lat"],
        "longitude":  site["lon"],
        "current":    "shortwave_radiation,wind_speed_10m,temperature_2m",
        "hourly":     "shortwave_radiation,wind_speed_10m",
        "forecast_days": 1,
        "timezone":   "UTC",
    }
    try:
        resp = await client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        hourly = data.get("hourly", {})

        radiation = current.get("shortwave_radiation")
        wind = current.get("wind_speed_10m")
        temp = current.get("temperature_2m")

        # Next 6 hourly values
        fc_radiation = (hourly.get("shortwave_radiation") or [])[:6]
        fc_wind = (hourly.get("wind_speed_10m") or [])[:6]

        return SiteWeather(
            duid=site["duid"],
            name=site["name"],
            region=site["region"],
            generator_type=site["type"],
            lat=site["lat"],
            lon=site["lon"],
            current_radiation_wm2=float(radiation) if radiation is not None else None,
            current_wind_kmh=float(wind) if wind is not None else None,
            current_temp_c=float(temp) if temp is not None else None,
            forecast_radiation=[round(float(v), 1) for v in fc_radiation if v is not None],
            forecast_wind=[round(float(v), 1) for v in fc_wind if v is not None],
        )
    except Exception as exc:
        logger.debug("OpenMeteo fetch failed for %s (%s): %s", site["duid"], site["name"], exc)
        return None


async def fetch_region_site_weather(region: str) -> RegionSiteWeather:
    """Fetch site-level solar irradiance + wind speed for all known generators in a region.

    Runs all site requests concurrently. Returns a RegionSiteWeather with
    whatever sites responded — never raises.
    """
    region = region.upper()
    sites = _SITES_BY_REGION.get(region, [])
    result = RegionSiteWeather(region=region)

    if not sites:
        return result

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            tasks = [_fetch_site_weather(client, s) for s in sites]
            raw = await asyncio.gather(*tasks, return_exceptions=True)

        for item in raw:
            if isinstance(item, SiteWeather):
                result.sites.append(item)
    except Exception as exc:
        logger.warning("OpenMeteo region fetch failed for %s: %s", region, exc)

    return result
