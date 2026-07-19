from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    CONF_RADAR_IMAGE_TYPE, RADAR_IMAGE_TYPE_CAPPI, DEFAULT_RADAR_IMAGE_TYPE,
    CONF_WEATHER_ENTITY, DEFAULT_WEATHER_ENTITY,
    CONF_USE_3D_WIND_PROFILE, DEFAULT_USE_3D_WIND_PROFILE,
)
from .open_meteo_client import get_3d_wind_profile
from .radar_processing import get_radar_info, get_forecast_info, get_precipitation_details


NOWCAST_SKILL_DECAY: dict[int, float] = {
    10: 0.95,
    20: 0.85,
    30: 0.70,
    40: 0.55,
    50: 0.40,
    60: 0.30,
}


@dataclass
class AladinRadar:
    rain_now: bool
    rain_now_value: float
    nearest_distance: float | None
    expected_rain_timestamp: datetime | None
    rain_probability: int
    rain_now_pixel_count: int
    forecast_coverages: dict[str, int]
    rain_duration_minutes: int
    exceeds_forecast_horizon: bool
    forecast_timeline: list[dict[str, Any]]
    precipitation_type: str | None = None
    freezing_level_m: float | None = None
    nwp_precipitation_probability: int | None = None


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
                minute=[2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57],
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
        if self._last_aladin_update is None or now.minute in (2, 7, 32, 37):
            LOGGER.debug("Updating Aladin weather data")
            try:
                await self.aladin_coordinator.async_refresh()
                self._last_aladin_update = now
            except Exception as ex:
                # Ošetření výjimky zajistí, že při výpadku Aladina nespadne update Radaru
                LOGGER.error("Error updating Aladin weather data: %s", ex)

        lat = self._config.get(CONF_LATITUDE, self.hass.config.latitude)
        lon = self._config.get(CONF_LONGITUDE, self.hass.config.longitude)
        session = aiohttp_client.async_get_clientsession(self.hass)

        options = self.config_entry.options
        use_3d_wind_profile = bool(options.get(CONF_USE_3D_WIND_PROFILE,
                                               self._config.get(CONF_USE_3D_WIND_PROFILE, DEFAULT_USE_3D_WIND_PROFILE)))
        radius = int(options.get(CONF_RADAR_RADIUS, self._config.get(CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS)))
        threshold_mmh = float(options.get(CONF_RADAR_THRESHOLD_MMH, DEFAULT_RADAR_THRESHOLD_MMH))
        window_size = int(options.get(CONF_RADAR_WINDOW_SIZE, DEFAULT_RADAR_WINDOW_SIZE))
        size_threshold = int(options.get(CONF_RADAR_SIZE_THRESHOLD, DEFAULT_RADAR_SIZE_THRESHOLD))
        image_type = options.get(CONF_RADAR_IMAGE_TYPE, DEFAULT_RADAR_IMAGE_TYPE)

        # Pure radar defaults: no sensor fusion unless user selects a weather entity
        humidity: float | None = None
        temp_c: float | None = None
        wind_speed_ms: float = 0.0
        wind_bearing_deg: int = 0
        profile_data = None
        nwp_precipitation_probability: int | None = None

        source_used = "pure_radar"

        if use_3d_wind_profile:
            wind_data = await get_3d_wind_profile(session, lat, lon)
            if wind_data:
                profile_data = wind_data["profile"]
                nwp_precipitation_probability = wind_data.get("precipitation_probability")
                source_used = "open_meteo_3d"
                # Extrakce povrchové vrstvy (nejnižší geopotenciální výška) pouze pro vizuální log
                surface = profile_data[-1]
                humidity = surface["humidity_pct"]
                temp_c = surface.get("temp_c")
                wind_speed_ms = surface["wind_speed_ms"]
                wind_bearing_deg = surface["wind_dir_deg"]
            else:
                LOGGER.warning("Open-Meteo 3D profile unavailable, falling back to 1D radar mode")
                source_used = "open_meteo_3d_unavailable"

        # Fallback se musí provést, pokud uživatel nemá zapnuté 3D, NEBO pokud 3D API selhalo
        if not profile_data:
            weather_entity_id = options.get(CONF_WEATHER_ENTITY,
                                            self._config.get(CONF_WEATHER_ENTITY, DEFAULT_WEATHER_ENTITY))

            if weather_entity_id:
                state = self.hass.states.get(weather_entity_id)
                if state is not None and state.state not in ("unavailable", "unknown"):
                    try:
                        raw_humidity = state.attributes.get("humidity")
                        raw_temperature = state.attributes.get("temperature")
                        raw_wind_speed = state.attributes.get("wind_speed")
                        raw_wind_bearing = state.attributes.get("wind_bearing")

                        if raw_humidity is not None:
                            humidity = float(raw_humidity)
                            source_used = f"entity:{weather_entity_id}"
                        else:
                            LOGGER.warning("Zdrojová entita %s neobsahuje atribut 'humidity', fallback na pure_radar",
                                           weather_entity_id)
                            source_used = f"entity:{weather_entity_id} (missing humidity)"

                        if raw_temperature is not None:
                            temp_c = float(raw_temperature)

                        if raw_wind_speed is not None:
                            wind_speed_ms = max(0.0, float(raw_wind_speed) / 3.6)

                        if raw_wind_bearing is not None:
                            wind_bearing_deg = int(raw_wind_bearing) % 360

                    except Exception as ex:
                        LOGGER.error("Chyba při parsování atributů z entity %s: %s", weather_entity_id, ex)
                        source_used = f"entity:{weather_entity_id} (parse error)"
                else:
                    LOGGER.debug("Zdrojová entita %s není dostupná (stav: %s).", weather_entity_id,
                                 state.state if state else "Not Found")

        # Final informational log about chosen fusion source
        LOGGER.debug(
            "Fetching radar data using mode: %s (temp=%s, humidity=%s, wind=%.1f m/s from %d°) - source=%s",
            image_type,
            f"{temp_c:.1f}°C" if temp_c is not None else "None",
            f"{humidity:.1f}%" if humidity is not None else "None (pure_radar)",
            wind_speed_ms,
            wind_bearing_deg,
            source_used,
        )

        # Zbytek radar logiky zůstává stejný. rounded_now se spočítá korektně.
        rounded_now = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)

        # 1. Stažení aktuálního snímku (s fallbackem o 5 minut zpět)
        radar_info: dict[str, Any] = {"rain_now": False, "rain_now_pixel_count": 0, "rain_now_value": 0.0,
                                      "nearest_distance": None}

        for offset in (0, 5):
            target_time = rounded_now - timedelta(minutes=offset)
            date_str = target_time.strftime("%Y%m%d")
            time_str = target_time.strftime("%H%M")

            if image_type == RADAR_IMAGE_TYPE_CAPPI:
                current_url = f"https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_cappi020/pacz2gmaps3.z_cappi020.{date_str}.{time_str}.0.png"
            else:  # RADAR_IMAGE_TYPE_MAX3D
                current_url = f"https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_max3d_masked/pacz2gmaps3.z_max3d.{date_str}.{time_str}.0.png"

            try:
                LOGGER.debug("Fetching current radar image (offset -%d min): %s", offset, current_url)
                response = await session.get(current_url)

                if response.status == HTTPStatus.OK:
                    image_bytes = await response.read()
                    radar_info = await self.hass.async_add_executor_job(
                        get_radar_info,
                        image_bytes,
                        lat,
                        lon,
                        radius,
                        threshold_mmh,
                        window_size,
                        size_threshold,
                        humidity,
                        wind_speed_ms,
                        wind_bearing_deg,
                        profile_data,
                        image_type,
                        temp_c
                    )
                    LOGGER.debug(
                        "Radar info for GPS [%s, %s]: rain_now=%s, rain_now_pixel_count=%s, nearest_distance=%s",
                        lat, lon, radar_info["rain_now"], radar_info["rain_now_pixel_count"],
                        radar_info["nearest_distance"]
                    )
                    break

                elif response.status == HTTPStatus.NOT_FOUND and offset == 0:
                    LOGGER.debug("Current radar image not found, attempting fallback.")
                    continue

            except Exception as ex:
                LOGGER.error("Error fetching current radar image: %s", ex)

        # 2. Stažení forecast snímků s fallbackem
        forecast_coverages: dict[str, int] = {}
        forecast_rain_states: dict[int, bool] = {}
        forecast_target_utc_map: dict[int, datetime] = {}
        forecast_timeline: list[dict[str, Any]] = []

        # Nowcasting model se počítá déle než samotný snímek, zkusíme aktuální, případně až 10 minut starý base
        for offset in (0, 5, 10):
            base_time_dt = rounded_now - timedelta(minutes=offset)
            base_date = base_time_dt.strftime("%Y%m%d")
            base_time = base_time_dt.strftime("%H%M")

            batch_ready = True
            first_fetched = True
            temp_coverages = {}
            temp_rain_states = {}
            temp_target_utc_map = {}
            temp_timeline = []

            for minutes in range(10, 70, 10):
                forecast_target = base_time_dt + timedelta(minutes=minutes)

                real_minutes_until = int((forecast_target - now).total_seconds() / 60)
                # Ignorujeme snímky, které zasahují do aktuálního stavu nebo minulosti
                if real_minutes_until < 5:
                    continue

                tgt_date = forecast_target.strftime("%Y%m%d")
                tgt_time = forecast_target.strftime("%H%M")

                # Sjednocená proměnná pro ukládání přesného času snímku do dictionary a senzoru
                forecast_target_utc = forecast_target.replace(tzinfo=timezone.utc)

                if image_type == RADAR_IMAGE_TYPE_CAPPI:
                    forecast_url = f"https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_cappi020_fct/{base_date}.{base_time}/pacz2gmaps3.fct_z_cappi020.{tgt_date}.{tgt_time}.{minutes}.png"
                else:  # RADAR_IMAGE_TYPE_MAX3D
                    forecast_url = f"https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_max3d_fct_masked/{base_date}.{base_time}/pacz2gmaps3.fct_z_max.{tgt_date}.{tgt_time}.{minutes}.png"

                try:
                    LOGGER.debug("Fetching forecast radar image (+%d min, base %s): %s", minutes, base_time,
                                 forecast_url)
                    response = await session.get(forecast_url)

                    if response.status == HTTPStatus.OK:
                        first_fetched = False
                        image_bytes = await response.read()

                        # Fetch comprehensive forecast info in a single pass
                        forecast_info = await self.hass.async_add_executor_job(
                            get_forecast_info,
                            image_bytes,
                            lat,
                            lon,
                            window_size,
                            threshold_mmh,
                            size_threshold,
                            humidity,
                            wind_speed_ms,
                            wind_bearing_deg,
                            profile_data,
                            image_type,
                            temp_c
                        )

                        cov = forecast_info["coverage_pct"]
                        has_rain = forecast_info["rain"]

                        temp_coverages[forecast_target_utc.isoformat()] = cov
                        temp_rain_states[real_minutes_until] = has_rain
                        temp_target_utc_map[real_minutes_until] = forecast_target_utc

                        temp_timeline.append({
                            "time": forecast_target_utc.isoformat(),
                            "rain": has_rain,
                            "coverage_pct": cov,
                            "intensity_mmh": forecast_info["intensity_mmh"],
                            "cloud_size_px": forecast_info["cloud_size_px"]
                        })

                    elif response.status == HTTPStatus.NOT_FOUND:
                        # If the first fetched image in this batch is missing, the batch is not ready yet
                        if first_fetched:
                            LOGGER.debug("Forecast batch %s not ready yet, falling back to older data.", base_time)
                            batch_ready = False
                            break  # Cancel the inner 10-60 loop and trigger the next offset iteration (-5 min)
                        first_fetched = False

                except Exception as ex:
                    LOGGER.debug("Error fetching forecast radar image for +%d min: %s", minutes, ex)

            # If we successfully processed the batch (at least the first image existed), save the data and end the fallback
            if batch_ready and temp_coverages:
                forecast_coverages = temp_coverages
                forecast_rain_states = temp_rain_states
                forecast_target_utc_map = temp_target_utc_map
                forecast_timeline = temp_timeline
                break

        # Calculate duration and expected timestamps based on collected states
        rain_duration_minutes = 0
        exceeds_forecast_horizon = False
        expected_rain_timestamp = None

        if radar_info["rain_now"]:
            # Find when it stops
            stopped_at_minute = None
            for m in sorted(forecast_rain_states.keys()):
                if not forecast_rain_states[m]:
                    stopped_at_minute = m
                    break

            if stopped_at_minute is None:
                # Never stopped in the available forecast data
                rain_duration_minutes = max(forecast_rain_states.keys()) if forecast_rain_states else 0
                exceeds_forecast_horizon = bool(forecast_rain_states)
                if exceeds_forecast_horizon:
                    LOGGER.debug("Rain is continuous beyond forecast horizon (>= %d min)", rain_duration_minutes)
            else:
                rain_duration_minutes = stopped_at_minute
                LOGGER.debug("Rain expected to stop in %d minutes", rain_duration_minutes)

                # Check if it starts again
                for m in sorted(forecast_rain_states.keys()):
                    if m > stopped_at_minute and forecast_rain_states[m]:
                        expected_rain_timestamp = forecast_target_utc_map[m]
                        LOGGER.debug("Next rain expected at %s (in %d minutes)", expected_rain_timestamp.isoformat(), m)
                        break
        else:
            # Not raining now, find when it starts
            for m in sorted(forecast_rain_states.keys()):
                if forecast_rain_states[m]:
                    expected_rain_timestamp = forecast_target_utc_map[m]
                    LOGGER.debug("Next rain expected at %s (in %d minutes)", expected_rain_timestamp.isoformat(), m)
                    break

        # Skill-decay radar probability:
        # Bereme maximální (coverage × skill_factor) přes celou timeline.
        # Pokud právě prší (rain_now), pravděpodobnost je 100%.
        if radar_info["rain_now"]:
            rain_probability = 100
        elif forecast_timeline:
            max_adjusted = 0.0
            for step in forecast_timeline:
                seconds = (datetime.fromisoformat(step["time"]) - now.replace(tzinfo=timezone.utc)).total_seconds()
                lead_min = max(10, min(60, int(round(seconds / 600.0) * 10)))
                skill = NOWCAST_SKILL_DECAY.get(lead_min, 0.30)
                adjusted = (step["coverage_pct"] / 100.0) * skill
                if adjusted > max_adjusted:
                    max_adjusted = adjusted
            rain_probability = round(max_adjusted * 100)
        else:
            rain_probability = 0

        precip_type, freezing_level = get_precipitation_details(profile_data)

        return AladinData(
            weather=self.aladin_coordinator.data,
            radar=AladinRadar(
                rain_now=radar_info["rain_now"],
                rain_now_value=radar_info["rain_now_value"],
                nearest_distance=radar_info["nearest_distance"],
                expected_rain_timestamp=expected_rain_timestamp,
                rain_probability=rain_probability,
                rain_now_pixel_count=radar_info["rain_now_pixel_count"],
                forecast_coverages=forecast_coverages,
                rain_duration_minutes=rain_duration_minutes,
                exceeds_forecast_horizon=exceeds_forecast_horizon,
                forecast_timeline=forecast_timeline,
                precipitation_type=precip_type,
                freezing_level_m=freezing_level,
                nwp_precipitation_probability=nwp_precipitation_probability,
            )
        )