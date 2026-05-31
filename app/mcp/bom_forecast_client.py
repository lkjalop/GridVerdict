"""BOM 7-day forecast client — daily weather outlook per NEM region.

Uses Open-Meteo's free daily forecast API (same provider already approved in
weather_client.py for current observations). Delivers the 7-day temperature,
wind speed, precipitation, and cloud cover outlook needed by:

  - A1: location + time horizon queries ("Brisbane next week" → QLD1 forecast)
  - B01: weather-price scenario queries ("showers Monday → price impact?")
  - B02: temperature spike scenarios ("40°C SA Thursday → pre-charge battery?")
  - LNN scenario planner: feed forward-looking weather instead of current obs

API: https://api.open-meteo.com/v1/forecast
     daily variables: temperature_2m_max, temperature_2m_min, wind_speed_10m_max,
                      precipitation_sum, cloud_cover_mean, shortwave_radiation_sum
     No API key required.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = httpx.Timeout(15.0)

# NEM region capital coordinates (matches weather_client.py REGION_WEATHER_POINTS)
_REGION_COORDS: dict[str, dict[str, Any]] = {
    "NSW1": {"name": "Sydney",    "lat": -33.8688, "lon": 151.2093, "tz": "Australia/Sydney"},
    "VIC1": {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631, "tz": "Australia/Melbourne"},
    "QLD1": {"name": "Brisbane",  "lat": -27.4698, "lon": 153.0251, "tz": "Australia/Brisbane"},
    "SA1":  {"name": "Adelaide",  "lat": -34.9285, "lon": 138.6007, "tz": "Australia/Adelaide"},
    "TAS1": {"name": "Hobart",    "lat": -42.8821, "lon": 147.3272, "tz": "Australia/Hobart"},
}

_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "precipitation_sum",
    "precipitation_probability_max",
    "cloud_cover_mean",
    "shortwave_radiation_sum",     # MJ/m² — proxy for solar irradiance / capacity factor
    "et0_fao_evapotranspiration",  # hydro reservoir inflow proxy
]


@dataclass
class DailyForecastDay:
    date: date
    temp_max_c: float | None
    temp_min_c: float | None
    wind_max_kmh: float | None
    wind_gust_kmh: float | None
    precip_mm: float | None
    precip_prob_pct: float | None
    cloud_pct: float | None
    solar_mj_m2: float | None    # shortwave radiation — solar generation proxy
    hydro_et0: float | None      # evapotranspiration — hydro inflow proxy

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "temp_max_c": self.temp_max_c,
            "temp_min_c": self.temp_min_c,
            "wind_max_kmh": self.wind_max_kmh,
            "wind_gust_kmh": self.wind_gust_kmh,
            "precip_mm": self.precip_mm,
            "precip_prob_pct": self.precip_prob_pct,
            "cloud_pct": self.cloud_pct,
            "solar_mj_m2": self.solar_mj_m2,
            "hydro_et0": self.hydro_et0,
        }

    # ── NEM-specific derived signals ────────────────────────────────────────

    @property
    def is_heatwave(self) -> bool:
        """True when max temperature exceeds 35°C (demand spike risk)."""
        return (self.temp_max_c or 0) >= 35.0

    @property
    def is_cold_snap(self) -> bool:
        """True when max temperature below 12°C (heating demand risk)."""
        return (self.temp_max_c or 99) < 12.0

    @property
    def solar_capacity_factor_proxy(self) -> float | None:
        """Rough solar CF estimate from daily radiation. 8 MJ/m² ≈ 0.25 CF."""
        if self.solar_mj_m2 is None:
            return None
        return min(self.solar_mj_m2 / 32.0, 0.95)   # 32 MJ/m² ≈ CF 1.0 theoretical max

    @property
    def wind_generation_factor(self) -> str:
        """Qualitative wind factor for LNN context injection."""
        kmh = self.wind_max_kmh or 0
        if kmh < 15:
            return "low"
        if kmh < 35:
            return "moderate"
        return "high"

    @property
    def demand_pressure(self) -> str:
        """Qualitative demand pressure: high on hot days, moderate on cold, low otherwise."""
        if self.is_heatwave:
            return "high_cooling"
        if self.is_cold_snap:
            return "elevated_heating"
        return "normal"


@dataclass
class RegionForecast:
    region: str
    city: str
    forecast_generated_at: datetime
    days: list[DailyForecastDay] = field(default_factory=list)
    raw_ref: str = ""
    source: str = "open-meteo.com"

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "city": self.city,
            "forecast_generated_at": self.forecast_generated_at.isoformat(),
            "source": self.source,
            "raw_ref": self.raw_ref,
            "days": [d.to_dict() for d in self.days],
        }

    def day_by_date(self, target: date) -> DailyForecastDay | None:
        for d in self.days:
            if d.date == target:
                return d
        return None

    def days_with_heatwave(self) -> list[DailyForecastDay]:
        return [d for d in self.days if d.is_heatwave]

    def peak_temp_day(self) -> DailyForecastDay | None:
        if not self.days:
            return None
        return max(self.days, key=lambda d: d.temp_max_c or -999)

    def lnn_feature_sequence(self) -> list[dict[str, Any]]:
        """Returns a list of daily feature dicts suitable for LNN scenario injection."""
        return [
            {
                "date": d.date.isoformat(),
                "temp_max_c": d.temp_max_c,
                "temp_min_c": d.temp_min_c,
                "wind_speed_kmh": d.wind_max_kmh,
                "cloud_cover_pct": d.cloud_pct,
                "precip_mm": d.precip_mm,
                "solar_cf_proxy": d.solar_capacity_factor_proxy,
                "demand_pressure": d.demand_pressure,
                "wind_factor": d.wind_generation_factor,
            }
            for d in self.days
        ]


async def fetch_7day_forecast(region: str) -> RegionForecast | None:
    """Fetch 7-day daily forecast for a NEM region. Returns None on failure."""
    region = region.upper()
    coords = _REGION_COORDS.get(region)
    if coords is None:
        logger.warning("Unknown region for forecast: %s", region)
        return None

    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "daily": ",".join(_DAILY_VARS),
        "timezone": coords["tz"],
        "forecast_days": 7,
        "wind_speed_unit": "kmh",
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_OPEN_METEO_FORECAST, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo 7-day forecast failed for %s: %s", region, exc)
        return None

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        return None

    def _get(key: str, i: int) -> float | None:
        vals = daily.get(key, [])
        v = vals[i] if i < len(vals) else None
        return float(v) if v is not None else None

    days: list[DailyForecastDay] = []
    for i, date_str in enumerate(dates):
        try:
            d = date.fromisoformat(date_str)
        except ValueError:
            continue
        days.append(DailyForecastDay(
            date=d,
            temp_max_c=_get("temperature_2m_max", i),
            temp_min_c=_get("temperature_2m_min", i),
            wind_max_kmh=_get("wind_speed_10m_max", i),
            wind_gust_kmh=_get("wind_gusts_10m_max", i),
            precip_mm=_get("precipitation_sum", i),
            precip_prob_pct=_get("precipitation_probability_max", i),
            cloud_pct=_get("cloud_cover_mean", i),
            solar_mj_m2=_get("shortwave_radiation_sum", i),
            hydro_et0=_get("et0_fao_evapotranspiration", i),
        ))

    raw_ref = (
        f"open-meteo.com/v1/forecast?lat={coords['lat']}&lon={coords['lon']}"
        f"&daily=temperature_2m_max,...&timezone={coords['tz']}&forecast_days=7"
    )
    return RegionForecast(
        region=region,
        city=coords["name"],
        forecast_generated_at=datetime.now(timezone.utc),
        days=days,
        raw_ref=raw_ref,
    )


async def fetch_multi_region_forecast(
    regions: list[str] | None = None,
) -> dict[str, RegionForecast]:
    """Fetch 7-day forecasts for multiple NEM regions concurrently."""
    import asyncio
    if regions is None:
        regions = list(_REGION_COORDS.keys())

    results = await asyncio.gather(
        *[fetch_7day_forecast(r) for r in regions],
        return_exceptions=True,
    )
    out: dict[str, RegionForecast] = {}
    for region, result in zip(regions, results):
        if isinstance(result, RegionForecast):
            out[region] = result
        elif isinstance(result, Exception):
            logger.debug("Forecast fetch failed for %s: %s", region, result)
    return out


def forecast_to_scatter_context(forecast: RegionForecast) -> dict[str, Any]:
    """Flatten a RegionForecast to a dict suitable for scatter-gather GatherResult."""
    peak = forecast.peak_temp_day()
    heatwaves = forecast.days_with_heatwave()
    return {
        "region": forecast.region,
        "city": forecast.city,
        "source": forecast.source,
        "raw_ref": forecast.raw_ref,
        "as_of": forecast.forecast_generated_at.isoformat(),
        "days": [d.to_dict() for d in forecast.days],
        "peak_temp_day": peak.to_dict() if peak else None,
        "heatwave_days": len(heatwaves),
        "heatwave_dates": [d.date.isoformat() for d in heatwaves],
        "lnn_feature_sequence": forecast.lnn_feature_sequence(),
        "available": True,
    }
