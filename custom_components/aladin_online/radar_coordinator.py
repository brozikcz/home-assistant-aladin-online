from __future__ import annotations
from datetime import timedelta
from http import HTTPStatus
from typing import Any
from types import MappingProxyType
from dataclasses import dataclass

from homeassistant import core
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt

from .aladin_online import AladinOnlineCoordinator, AladinWeather
from .const import DOMAIN, LOGGER, CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS
from .radar_processing import get_radar_info, check_forecast_rain, calculate_forecast_probability


@dataclass
class AladinRadar:
    rain_now: bool
    rain_now_value: float
    nearest_distance: float | None
    minutes_until_rain: int | None
    rain_probability: int
    rain_now_pixel_count: int
    forecast_probabilities: dict[str, int]


@dataclass
class AladinData:
    weather: AladinWeather | None
    radar: AladinRadar | None


class AladinRadarCoordinator(DataUpdateCoordinator[AladinData]):
    def __init__(self, hass: core.HomeAssistant, config: MappingProxyType) -> None:
        super().__init__(
            hass,
            LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=5),
            update_method=self._async_update_data
        )
        self._config = config
        self.aladin_coordinator = AladinOnlineCoordinator(hass, config)
        self._last_aladin_update = None

    async def _async_update_data(self) -> AladinData:
        now = dt.utcnow()

        if self._last_aladin_update is None or now - self._last_aladin_update >= timedelta(minutes=30):
            LOGGER.debug("Updating Aladin weather data")
            await self.aladin_coordinator.async_refresh()
            self._last_aladin_update = now

        lat = self._config.get(CONF_LATITUDE, self.hass.config.latitude)
        lon = self._config.get(CONF_LONGITUDE, self.hass.config.longitude)
        radius = self._config.get(CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS)

        session = aiohttp_client.async_get_clientsession(self.hass)

        # Zaokrouhlení na 5min blok
        rounded_now = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)

        # 1. Stažení aktuálního snímku z_max3d_masked (s fallbackem o 5 minut zpět)
        radar_info: dict[str, Any] = {"rain_now": False, "rain_now_pixel_count": 0, "rain_now_value": 0.0, "nearest_distance": None}

        for offset in (0, 5):
            target_time = rounded_now - timedelta(minutes=offset)
            date_str = target_time.strftime("%Y%m%d")
            time_str = target_time.strftime("%H%M")

            current_url = f"https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_max3d_masked/pacz2gmaps3.z_max3d.{date_str}.{time_str}.0.png"

            try:
                LOGGER.debug("Fetching current radar image (offset -%d min): %s", offset, current_url)
                response = await session.get(current_url)

                if response.status == HTTPStatus.OK:
                    image_bytes = await response.read()
                    radar_info = await self.hass.async_add_executor_job(get_radar_info, image_bytes, lat, lon, radius)
                    LOGGER.debug(
                        "Radar info for GPS [%s, %s]: rain_now=%s, rain_now_pixel_count=%s, nearest_distance_px=%s, nearest_pixel=%s, nearest_gps=%s",
                        lat, lon, radar_info["rain_now"], radar_info["rain_now_pixel_count"], radar_info["nearest_distance"],
                        radar_info.get("nearest_pixel"), radar_info.get("nearest_gps")
                    )
                    break

                elif response.status == HTTPStatus.NOT_FOUND and offset == 0:
                    LOGGER.debug("Current radar image not found, attempting fallback.")
                    continue

            except Exception as ex:
                LOGGER.error("Error fetching current radar image: %s", ex)

        # 2. Stažení forecast snímků z_max3d_fct_masked
        # 2. Stažení forecast snímků z_max3d_fct_masked s fallbackem
        minutes_until_rain = None
        forecast_probabilities: dict[str, int] = {}

        # Nowcasting model se počítá déle než samotný snímek, zkusíme aktuální, případně až 10 minut starý base
        for offset in (0, 5, 10):
            base_time_dt = rounded_now - timedelta(minutes=offset)
            base_date = base_time_dt.strftime("%Y%m%d")
            base_time = base_time_dt.strftime("%H%M")

            batch_ready = True
            temp_probabilities = {}
            temp_minutes = None

            for minutes in range(10, 70, 10):
                forecast_target = base_time_dt + timedelta(minutes=minutes)
                tgt_date = forecast_target.strftime("%Y%m%d")
                tgt_time = forecast_target.strftime("%H%M")

                forecast_url = f"https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_max3d_fct_masked/{base_date}.{base_time}/pacz2gmaps3.fct_z_max.{tgt_date}.{tgt_time}.{minutes}.png"

                try:
                    LOGGER.debug("Fetching forecast radar image (+%d min, base %s): %s", minutes, base_time,
                                 forecast_url)
                    response = await session.get(forecast_url)

                    if response.status == HTTPStatus.OK:
                        image_bytes = await response.read()

                        prob = await self.hass.async_add_executor_job(calculate_forecast_probability,
                                                                      image_bytes, lat, lon)
                        temp_probabilities[f"{minutes}min"] = prob

                        if temp_minutes is None:
                            has_rain = await self.hass.async_add_executor_job(check_forecast_rain, image_bytes,
                                                                              lat, lon)
                            if has_rain:
                                LOGGER.info("Significant rain forecasted in %d minutes", minutes)
                                temp_minutes = minutes

                    elif response.status == HTTPStatus.NOT_FOUND:
                        # Pokud chybí hned první snímek, nemá smysl zkoušet zbytek sady
                        if minutes == 10:
                            LOGGER.debug("Forecast batch %s not ready yet, falling back to older data.",
                                         base_time)
                            batch_ready = False
                            break  # Zruší vnitřní smyčku 10-60 a vyvolá další iteraci offsetu (-5 min)

                except Exception as ex:
                    LOGGER.debug("Error fetching forecast radar image for +%d min: %s", minutes, ex)

            # Pokud jsme úspěšně prošli dávku (alespoň první snímek existoval), uložíme data a končíme fallback
            if batch_ready and temp_probabilities:
                forecast_probabilities = temp_probabilities
                minutes_until_rain = temp_minutes
                break

        rain_probability = max(forecast_probabilities.values()) if forecast_probabilities else 0

        return AladinData(
            weather=self.aladin_coordinator.data,
            radar=AladinRadar(
                rain_now=radar_info["rain_now"],
                rain_now_value=radar_info["rain_now_value"],
                nearest_distance=radar_info["nearest_distance"],
                minutes_until_rain=minutes_until_rain,
                rain_probability=rain_probability,
                rain_now_pixel_count=radar_info["rain_now_pixel_count"],
                forecast_probabilities=forecast_probabilities,
            )
        )