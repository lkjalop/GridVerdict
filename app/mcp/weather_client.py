"""Read-only weather consensus client for NEM market context.

Weather is treated as contextual evidence only. A high temperature, wind lull,
or cloud cover observation may support a market explanation, but GridVerdict
must not claim weather caused a price move unless the user asked for weather
context and other market evidence also supports the claim.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any

import httpx


REGION_WEATHER_POINTS: dict[str, dict[str, Any]] = {
    "NSW1": {
        "name": "Sydney",
        "lat": -33.8688,
        "lon": 151.2093,
        "bom_product": "IDN60920",
    },
    "VIC1": {
        "name": "Melbourne",
        "lat": -37.8136,
        "lon": 144.9631,
        "bom_product": "IDV60920",
    },
    "QLD1": {
        "name": "Brisbane",
        "lat": -27.4698,
        "lon": 153.0251,
        "bom_product": "IDQ60920",
    },
    "SA1": {
        "name": "Adelaide",
        "lat": -34.9285,
        "lon": 138.6007,
        "bom_product": "IDS60920",
    },
    "TAS1": {
        "name": "Hobart",
        "lat": -42.8821,
        "lon": 147.3272,
        "bom_product": "IDT60920",
    },
}


@dataclass
class WeatherReading:
    source: str
    observed_at: datetime
    temperature_c: float | None = None
    humidity_pct: float | None = None
    wind_speed_kmh: float | None = None
    wind_gust_kmh: float | None = None
    precipitation_mm: float | None = None
    cloud_cover_pct: float | None = None
    raw_ref: str = ""
    caveat: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "observed_at": self.observed_at.isoformat(),
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "wind_speed_kmh": self.wind_speed_kmh,
            "wind_gust_kmh": self.wind_gust_kmh,
            "precipitation_mm": self.precipitation_mm,
            "cloud_cover_pct": self.cloud_cover_pct,
            "raw_ref": self.raw_ref,
            "caveat": self.caveat,
        }


class WeatherConsensusClient:
    """Fetch and reconcile weather from three public, read-only sources."""

    def __init__(self, timeout_s: float = 8.0) -> None:
        self.timeout = httpx.Timeout(timeout_s)

    async def fetch_region_consensus(self, region: str) -> dict[str, Any]:
        region = region.upper()
        point = REGION_WEATHER_POINTS.get(region, REGION_WEATHER_POINTS["NSW1"])
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            tasks = [
                self._fetch_open_meteo(client, point),
                self._fetch_met_no(client, point),
                self._fetch_bom_observation(client, point),
            ]
            import asyncio

            results = await asyncio.gather(*tasks, return_exceptions=True)

        readings = [r for r in results if isinstance(r, WeatherReading)]
        errors = [
            {"source": _source_name(i), "error": str(r)}
            for i, r in enumerate(results)
            if isinstance(r, Exception)
        ]
        return build_weather_consensus(region, point, readings, errors)

    async def _fetch_open_meteo(self, client: httpx.AsyncClient, point: dict[str, Any]) -> WeatherReading:
        params = {
            "latitude": point["lat"],
            "longitude": point["lon"],
            "current": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
                "wind_gusts_10m",
            ]),
            "timezone": "Australia/Sydney",
        }
        url = "https://api.open-meteo.com/v1/forecast"
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        cur = data.get("current") or {}
        observed = _parse_dt(cur.get("time"))
        return WeatherReading(
            source="OPEN_METEO",
            observed_at=observed,
            temperature_c=_num(cur.get("temperature_2m")),
            humidity_pct=_num(cur.get("relative_humidity_2m")),
            wind_speed_kmh=_num(cur.get("wind_speed_10m")),
            wind_gust_kmh=_num(cur.get("wind_gusts_10m")),
            precipitation_mm=_num(cur.get("precipitation")),
            cloud_cover_pct=_num(cur.get("cloud_cover")),
            raw_ref=str(resp.url),
            caveat="Model-derived current conditions; useful for consensus, not observed station truth.",
        )

    async def _fetch_met_no(self, client: httpx.AsyncClient, point: dict[str, Any]) -> WeatherReading:
        url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
        headers = {"User-Agent": "GridVerdict/0.1 contact:local"}
        resp = await client.get(
            url,
            params={"lat": point["lat"], "lon": point["lon"]},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        row = (data.get("properties", {}).get("timeseries") or [{}])[0]
        instant = row.get("data", {}).get("instant", {}).get("details", {})
        next_hour = row.get("data", {}).get("next_1_hours", {}).get("details", {})
        observed = _parse_dt(row.get("time"))
        wind_mps = _num(instant.get("wind_speed"))
        return WeatherReading(
            source="MET_NO",
            observed_at=observed,
            temperature_c=_num(instant.get("air_temperature")),
            humidity_pct=_num(instant.get("relative_humidity")),
            wind_speed_kmh=round(wind_mps * 3.6, 2) if wind_mps is not None else None,
            precipitation_mm=_num(next_hour.get("precipitation_amount")),
            cloud_cover_pct=_num(instant.get("cloud_area_fraction")),
            raw_ref=str(resp.url),
            caveat="Global point forecast from MET Norway; model evidence, not local station observation.",
        )

    async def _fetch_bom_observation(self, client: httpx.AsyncClient, point: dict[str, Any]) -> WeatherReading:
        product = point["bom_product"]
        url = f"http://www.bom.gov.au/fwo/{product}.xml"
        resp = await client.get(url, headers={"User-Agent": "GridVerdict/0.1"})
        resp.raise_for_status()
        if len(resp.content) > 2_000_000:
            raise ValueError("BOM observation XML too large")
        return _parse_bom_xml(resp.text, point, raw_ref=url)


def build_weather_consensus(
    region: str,
    point: dict[str, Any],
    readings: list[WeatherReading],
    errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    fields = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_kmh",
        "wind_gust_kmh",
        "precipitation_mm",
        "cloud_cover_pct",
    ]
    consensus: dict[str, Any] = {}
    spread: dict[str, float] = {}
    source_count: dict[str, int] = {}
    for field in fields:
        vals = [getattr(r, field) for r in readings if getattr(r, field) is not None]
        source_count[field] = len(vals)
        if vals:
            consensus[field] = round(float(median(vals)), 2)
            spread[field] = round(float(max(vals) - min(vals)), 2)
        else:
            consensus[field] = None
            spread[field] = 0.0

    confidence = _consensus_confidence(readings, spread)
    tags = _weather_tags(consensus)
    return {
        "source": "WEATHER_CONSENSUS",
        "region": region,
        "location": point["name"],
        "lat": point["lat"],
        "lon": point["lon"],
        "observed_at": max((r.observed_at for r in readings), default=datetime.now(timezone.utc)).isoformat(),
        "consensus": consensus,
        "spread": spread,
        "source_count": source_count,
        "confidence": confidence,
        "relevance_tags": tags,
        "readings": [r.to_dict() for r in readings],
        "errors": errors or [],
        "raw_ref": ",".join(r.raw_ref for r in readings if r.raw_ref),
        "caveat": (
            "Weather consensus compares observed BOM station data with model sources. "
            "It is contextual evidence for demand/renewables, not a standalone price cause."
        ),
    }


def weather_query_relevant(
    text: str,
    region: str | None = None,
    regime: str | None = None,
) -> bool:
    """Return True when weather context should be fetched for this query.

    Three triggers:
      1. Query keywords — explicit weather/demand/renewable language.
      2. Regime-driven — elevated/spike/extreme in warm-climate regions (SA1, QLD1, VIC1, NSW1)
         where temperature is a known demand driver.
      3. Season-driven — Australian summer (Nov–Mar) in SA1/QLD1/VIC1, when heat-load demand
         spikes are common regardless of query wording.
    """
    lower = text.lower()
    keywords = [
        "weather", "temperature", "heat", "hot", "cold",
        "wind", "solar", "cloud", "rain", "storm", "humidity",
        "demand", "renewable", "rooftop",
    ]
    if any(k in lower for k in keywords):
        return True
    # Regime-driven: elevated/spike regimes in temperature-sensitive regions
    if regime in ("elevated", "spike", "extreme") and region in ("SA1", "QLD1", "VIC1", "NSW1"):
        return True
    # Season-driven: Australian summer months in hot regions
    from datetime import datetime, timezone
    month = datetime.now(timezone.utc).month
    if month in (11, 12, 1, 2, 3) and region in ("SA1", "QLD1", "VIC1"):
        return True
    return False


def _parse_bom_xml(xml_text: str, point: dict[str, Any], raw_ref: str) -> WeatherReading:
    root = ET.fromstring(xml_text)
    stations = root.findall(".//station")
    best_station = None
    best_dist = float("inf")
    for station in stations:
        try:
            lat = float(station.attrib.get("lat"))
            lon = float(station.attrib.get("lon"))
        except (TypeError, ValueError):
            continue
        dist = _distance_km(point["lat"], point["lon"], lat, lon)
        if dist < best_dist:
            best_dist = dist
            best_station = station
    if best_station is None:
        raise ValueError("No BOM station with coordinates found")

    period = best_station.find(".//period")
    level = period.find(".//level") if period is not None else None
    if level is None:
        raise ValueError("No BOM observation level found")

    values: dict[str, float | None] = {}
    for el in level.findall("element"):
        typ = el.attrib.get("type", "")
        values[typ] = _num(el.text)
    observed = _parse_dt(period.attrib.get("time-utc") if period is not None else None)
    return WeatherReading(
        source="BOM_OBSERVATION",
        observed_at=observed,
        temperature_c=values.get("air_temperature"),
        humidity_pct=values.get("rel-humidity"),
        wind_speed_kmh=values.get("wind_spd_kmh"),
        wind_gust_kmh=values.get("gust_kmh"),
        precipitation_mm=values.get("rain_trace"),
        raw_ref=f"{raw_ref}#{best_station.attrib.get('bom-id', best_station.attrib.get('stn-name', 'station'))}",
        caveat=f"Nearest BOM observation station to {point['name']} ({best_dist:.0f} km).",
    )


def _weather_tags(consensus: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    temp = consensus.get("temperature_c")
    wind = consensus.get("wind_speed_kmh")
    cloud = consensus.get("cloud_cover_pct")
    rain = consensus.get("precipitation_mm")
    if temp is not None and temp >= 32:
        tags.append("heat_load_risk")
    if temp is not None and temp <= 8:
        tags.append("cold_load_risk")
    if wind is not None and wind <= 10:
        tags.append("low_wind_risk")
    if wind is not None and wind >= 45:
        tags.append("high_wind_or_storm_risk")
    if cloud is not None and cloud >= 75:
        tags.append("low_solar_risk")
    if rain is not None and rain > 0:
        tags.append("rain_or_storm_context")
    return tags


def _consensus_confidence(readings: list[WeatherReading], spread: dict[str, float]) -> float:
    if not readings:
        return 0.0
    score = min(len(readings), 3) / 3.0
    if spread.get("temperature_c", 0.0) > 5:
        score -= 0.2
    if spread.get("wind_speed_kmh", 0.0) > 25:
        score -= 0.2
    if not any(r.source == "BOM_OBSERVATION" for r in readings):
        score -= 0.15
    return round(max(0.0, min(1.0, score)), 3)


def _source_name(idx: int) -> str:
    return ["OPEN_METEO", "MET_NO", "BOM_OBSERVATION"][idx] if idx < 3 else "UNKNOWN"


def _num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    )
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))
