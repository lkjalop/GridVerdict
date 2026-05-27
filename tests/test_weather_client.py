from datetime import datetime, timezone

from app.mcp.weather_client import WeatherReading, build_weather_consensus, weather_query_relevant


def test_weather_query_relevance_detects_weather_and_renewables():
    assert weather_query_relevant("Is wind causing SA prices to spike?")
    assert weather_query_relevant("Will hot weather increase demand in NSW?")
    assert not weather_query_relevant("What is the current NSW price?")


def test_weather_consensus_uses_median_and_spread():
    point = {"name": "Sydney", "lat": -33.86, "lon": 151.2}
    now = datetime(2026, 5, 24, tzinfo=timezone.utc)
    readings = [
        WeatherReading("BOM_OBSERVATION", now, temperature_c=35.0, wind_speed_kmh=8.0, raw_ref="bom"),
        WeatherReading("OPEN_METEO", now, temperature_c=33.0, wind_speed_kmh=12.0, raw_ref="om"),
        WeatherReading("MET_NO", now, temperature_c=34.0, wind_speed_kmh=10.0, raw_ref="met"),
    ]

    result = build_weather_consensus("NSW1", point, readings)

    assert result["consensus"]["temperature_c"] == 34.0
    assert result["consensus"]["wind_speed_kmh"] == 10.0
    assert result["spread"]["temperature_c"] == 2.0
    assert result["confidence"] == 1.0
    assert "heat_load_risk" in result["relevance_tags"]
    assert "low_wind_risk" in result["relevance_tags"]


def test_weather_consensus_degrades_without_bom():
    point = {"name": "Sydney", "lat": -33.86, "lon": 151.2}
    now = datetime(2026, 5, 24, tzinfo=timezone.utc)
    readings = [
        WeatherReading("OPEN_METEO", now, temperature_c=20.0, wind_speed_kmh=20.0),
        WeatherReading("MET_NO", now, temperature_c=26.0, wind_speed_kmh=50.0),
    ]

    result = build_weather_consensus("NSW1", point, readings)

    assert result["confidence"] < 0.7
    assert result["source_count"]["temperature_c"] == 2
