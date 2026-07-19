from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp

LOGGER = logging.getLogger(__name__)

_OPEN_METEO_CACHE = {"timestamp": 0.0, "data": None}
CACHE_TTL_SECONDS = 900


async def get_3d_wind_profile(session: aiohttp.ClientSession, lat: float, lon: float) -> dict[str, Any] | None:
    """Fetch 3D wind and humidity profile from Open-Meteo with 15-minute caching."""
    global _OPEN_METEO_CACHE

    current_time = time.time()

    # Kontrola platnosti cache
    if _OPEN_METEO_CACHE["data"] is not None and (current_time - _OPEN_METEO_CACHE["timestamp"]) < CACHE_TTL_SECONDS:
        age = current_time - _OPEN_METEO_CACHE["timestamp"]
        LOGGER.debug("Open-Meteo: Using cached 3D profile data (cache age: %.1f s)", age)
        return extract_current_hour_profile(_OPEN_METEO_CACHE["data"])

    LOGGER.debug("Open-Meteo: Cache expired or empty. Downloading fresh 3D profile data for lat=%s, lon=%s...", lat, lon)
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&hourly=precipitation_probability,temperature_800hPa,temperature_850hPa,temperature_900hPa,temperature_925hPa,temperature_950hPa,temperature_975hPa,"
        "relative_humidity_800hPa,relative_humidity_850hPa,relative_humidity_900hPa,relative_humidity_925hPa,relative_humidity_950hPa,relative_humidity_975hPa,"
        "wind_speed_800hPa,wind_speed_850hPa,wind_speed_900hPa,wind_speed_925hPa,wind_speed_950hPa,wind_speed_975hPa,"
        "wind_direction_800hPa,wind_direction_850hPa,wind_direction_900hPa,wind_direction_925hPa,wind_direction_950hPa,wind_direction_975hPa,"
        "geopotential_height_800hPa,geopotential_height_850hPa,geopotential_height_900hPa,geopotential_height_925hPa,geopotential_height_950hPa,geopotential_height_975hPa"
        "&models=icon_d2&forecast_days=1&timezone=UTC"
    )

    try:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                _OPEN_METEO_CACHE["timestamp"] = current_time
                _OPEN_METEO_CACHE["data"] = data
                LOGGER.debug("Open-Meteo: Successfully fetched and cached new 3D profile")
                return extract_current_hour_profile(data)

            LOGGER.error("Open-Meteo API error: HTTP %s", response.status)
            return None
    except Exception as ex:
        LOGGER.error("Open-Meteo connection failed: %s", ex)
        return None


def extract_current_hour_profile(data: dict[str, Any]) -> dict[str, Any] | None:
    """Parses API response and returns profile layers + NWP probability for the current UTC hour."""
    try:
        hourly = data["hourly"]
        times = hourly["time"]
        current_utc_iso = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")

        try:
            idx = times.index(current_utc_iso)
        except ValueError:
            idx = 0

        levels = ["800hPa", "850hPa", "900hPa", "925hPa", "950hPa", "975hPa"]
        profile = []

        for lvl in levels:
            profile.append(
                {
                    "level": lvl,
                    "height_m": hourly[f"geopotential_height_{lvl}"][idx],
                    "wind_speed_ms": hourly[f"wind_speed_{lvl}"][idx] / 3.6,
                    "wind_dir_deg": hourly[f"wind_direction_{lvl}"][idx],
                    "humidity_pct": hourly[f"relative_humidity_{lvl}"][idx],
                    "temp_c": hourly[f"temperature_{lvl}"][idx],
                }
            )

        precip_prob_key = "precipitation_probability"
        nwp_precip_probability = hourly.get(precip_prob_key, [None] * len(times))[idx] if precip_prob_key in hourly else None

        return {
            "profile": sorted(profile, key=lambda x: x["height_m"], reverse=True),
            "precipitation_probability": nwp_precip_probability,
        }
    except KeyError as ex:
        LOGGER.error("Malformed Open-Meteo data: missing %s", ex)
        return None
