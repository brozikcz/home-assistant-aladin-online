from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from http import HTTPStatus
from typing import Any

from homeassistant import core
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt
from .aladin_online import AladinOnlineCoordinator, AladinWeather
from .const import (
    DOMAIN, LOGGER, CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS,
    CONF_RADAR_THRESHOLD_MMH, DEFAULT_RADAR_THRESHOLD_MMH,
    CONF_RADAR_WINDOW_SIZE, DEFAULT_RADAR_WINDOW_SIZE,
    CONF_RADAR_SIZE_THRESHOLD, DEFAULT_RADAR_SIZE_THRESHOLD,
)
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
    def __init__(self, hass: core.HomeAssistant, config_entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            LOGGER,
            config_entry=config_entry,
            name=DOMAIN,
            update_interval=None,
            update_method=self._async_update_data
        )
        self._config = config_entry.data
        self.aladin_coordinator = AladinOnlineCoordinator(hass, config_entry.data)
        self._last_aladin_update = None

        # Fixní cron-like trigger pro spuštění v minutách: 1, 6, 11, 16, atd.
        # async_on_unload zajistí odstranění časovače při odebrání integrace
        config_entry.async_on_unload(
            async_track_time_change(
                self.hass,
                self._handle_timer,
                minute=[1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56],
                second=0
            )
        )

    async def _handle_timer(self, now: dt.datetime) -> None:
        """Callback vyvolaný přesně podle async_track_time_pattern."""
        await self.async_request_refresh()

    async def _async_update_data(self) -> AladinData:
        now = dt.utcnow()

        # Aladin update - spouštíme jen v definovaných minutách (01, 06, 31, 36)
        # nebo pokud integrace ještě nemá data.
        if self._last_aladin_update is None or now.minute in (1, 6, 31, 36):
            LOGGER.debug("Updating Aladin weather data")
            try:
                await self.aladin_coordinator.async_refresh()
                self._last_aladin_update = now
            except Exception as ex:
                # Ošetření výjimky zajistí, že při výpadku Aladina nespadne update Radaru
                LOGGER.error("Error updating Aladin weather data: %s", ex)

        lat = self._config.get(CONF_LATITUDE, self.hass.config.latitude)
        lon = self._config.get(CONF_LONGITUDE, self.hass.config.longitude)

        options = self.config_entry.options
        radius = int(options.get(CONF_RADAR_RADIUS, self._config.get(CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS)))
        threshold_mmh = float(options.get(CONF_RADAR_THRESHOLD_MMH, DEFAULT_RADAR_THRESHOLD_MMH))
        window_size = int(options.get(CONF_RADAR_WINDOW_SIZE, DEFAULT_RADAR_WINDOW_SIZE))
        size_threshold = int(options.get(CONF_RADAR_SIZE_THRESHOLD, DEFAULT_RADAR_SIZE_THRESHOLD))

        session = aiohttp_client.async_get_clientsession(self.hass)

        # Zbytek radar logiky zůstává stejný. rounded_now se spočítá korektně.
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
                    radar_info = await self.hass.async_add_executor_job(get_radar_info, image_bytes, lat, lon, radius, threshold_mmh, window_size, size_threshold)
                    LOGGER.debug(
                        "Radar info for GPS [%s, %s]: rain_now=%s, rain_now_pixel_count=%s, nearest_distance=%s",
                        lat, lon, radar_info["rain_now"], radar_info["rain_now_pixel_count"], radar_info["nearest_distance"]
                    )
                    break

                elif response.status == HTTPStatus.NOT_FOUND and offset == 0:
                    LOGGER.debug("Current radar image not found, attempting fallback.")
                    continue

            except Exception as ex:
                LOGGER.error("Error fetching current radar image: %s", ex)

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

                        # Sjednocení parametrů citlivosti pro výpočet pravděpodobnosti
                        prob = await self.hass.async_add_executor_job(
                            calculate_forecast_probability,
                            image_bytes,
                            lat,
                            lon,
                            window_size,
                            threshold_mmh
                        )
                        temp_probabilities[f"{minutes}min"] = prob

                        if temp_minutes is None:
                            has_rain = await self.hass.async_add_executor_job(check_forecast_rain, image_bytes,
                                                                              lat, lon, threshold_mmh, window_size, size_threshold)
                            if has_rain:
                                LOGGER.debug("Significant rain forecasted in %d minutes", minutes)
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

        # Zahrnutí aktuálního deště (rain_now = 100% pravděpodobnost srážek) do celkové pravděpodobnosti
        current_rain_prob = 100 if radar_info["rain_now"] else 0
        all_probabilities = list(forecast_probabilities.values()) + [current_rain_prob]
        rain_probability = max(all_probabilities) if all_probabilities else 0

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