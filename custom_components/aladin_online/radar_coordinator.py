from __future__ import annotations
from datetime import timedelta
from http import HTTPStatus
from typing import Final
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
	forecast_probabilities: dict[str, int]

@dataclass
class AladinData:
	weather: AladinWeather | None
	radar: AladinRadar | None

class AladinRadarCoordinator(DataUpdateCoordinator[AladinData]):
	def __init__(self, hass: core.HomeAssistant, config: MappingProxyType) -> None:
		super().__init__(hass, LOGGER, name=DOMAIN, update_interval=timedelta(minutes=5), update_method=self._async_update_data)
		self._config = config
		self.aladin_coordinator = AladinOnlineCoordinator(hass, config)
		self._last_aladin_update = None

	async def _async_update_data(self) -> AladinData:
		# Update Aladin data every 30 minutes
		if self._last_aladin_update is None or dt.utcnow() - self._last_aladin_update >= timedelta(minutes=30):
			LOGGER.debug("Updating Aladin weather data")
			await self.aladin_coordinator.async_refresh()
			self._last_aladin_update = dt.utcnow()

		lat = self._config.get(CONF_LATITUDE, self.hass.config.latitude)
		lon = self._config.get(CONF_LONGITUDE, self.hass.config.longitude)
		radius = self._config.get(CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS)
		
		session = aiohttp_client.async_get_clientsession(self.hass)
		now = dt.utcnow()
		
		# Round time to 5 minutes as CHMI updates every 5-10 mins
		rounded_now = now - timedelta(minutes=now.minute % 5, seconds=now.second, microseconds=now.microsecond)
		
		current_url = "https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_max3d_masked/pacz2gmaps3.z_max3d.{}.{}.0.png".format(
			rounded_now.strftime("%Y%m%d"),
			rounded_now.strftime("%H%M")
		)
		
		radar_info = {"rain_now": False, "rain_now_value": 0.0, "nearest_distance": None}
		try:
			LOGGER.debug("Fetching current radar image: %s", current_url)
			response = await session.get(current_url)
			if response.status == HTTPStatus.OK:
				image_bytes = await response.read()
				radar_info = await self.hass.async_add_executor_job(get_radar_info, image_bytes, lat, lon, radius)
				LOGGER.debug("Radar info for GPS [%s, %s]: rain_now=%s, rain_now_value=%s, nearest_distance=%s", lat, lon, radar_info["rain_now"], radar_info["rain_now_value"], radar_info["nearest_distance"])
			elif response.status == HTTPStatus.NOT_FOUND:
				# Try 5 minutes older if not found
				rounded_now -= timedelta(minutes=5)
				current_url = "https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_max3d_masked/pacz2gmaps3.z_max3d.{}.{}.0.png".format(
					rounded_now.strftime("%Y%m%d"),
					rounded_now.strftime("%H%M")
				)
				LOGGER.debug("Radar image not found, trying 5 minutes older: %s", current_url)
				response = await session.get(current_url)
				if response.status == HTTPStatus.OK:
					image_bytes = await response.read()
					radar_info = await self.hass.async_add_executor_job(get_radar_info, image_bytes, lat, lon, radius)
					LOGGER.debug("Radar info (fallback): rain_now=%s, rain_now_value=%s, nearest_distance=%s", radar_info["rain_now"], radar_info["rain_now_value"], radar_info["nearest_distance"])
		except Exception as ex:
			LOGGER.error("Error fetching current radar image: %s", ex)

		minutes_until_rain = None
		forecast_probabilities: dict[str, int] = {}
		# Forecasted rain URL
		# Scan +10, +20, +30, +40, +50, +60 minutes
		for minutes in range(10, 70, 10):
			target_time = rounded_now + timedelta(minutes=minutes)
			forecast_url = "https://intranet.chmi.cz/files/portal/docs/meteo/rad/inca-cz/data/czrad-z_max3d_fct_masked/{}.{}/pacz2gmaps3.fct_z_max.{}.{}.{}.png".format(
				rounded_now.strftime("%Y%m%d"),
				rounded_now.strftime("%H%M"),
				target_time.strftime("%Y%m%d"),
				target_time.strftime("%H%M"),
				minutes
			)
			try:
				LOGGER.debug("Fetching forecast radar image (+%d min): %s", minutes, forecast_url)
				response = await session.get(forecast_url)
				if response.status == HTTPStatus.OK:
					image_bytes = await response.read()
					
					# Calculate probability in 7x7 window
					prob = await self.hass.async_add_executor_job(calculate_forecast_probability, image_bytes, lat, lon)
					forecast_probabilities[f"{minutes}min"] = prob
					
					if minutes_until_rain is None and await self.hass.async_add_executor_job(check_forecast_rain, image_bytes, lat, lon):
						LOGGER.info("Rain forecasted in %d minutes", minutes)
						minutes_until_rain = minutes
			except Exception as ex:
				LOGGER.debug("Error fetching forecast radar image for +%d min: %s", minutes, ex)

		rain_probability = max(forecast_probabilities.values()) if forecast_probabilities else 0

		return AladinData(
			weather=self.aladin_coordinator.data,
			radar=AladinRadar(
				rain_now=radar_info["rain_now"],
				rain_now_value=radar_info["rain_now_value"],
				nearest_distance=radar_info["nearest_distance"],
				minutes_until_rain=minutes_until_rain,
				rain_probability=rain_probability,
				forecast_probabilities=forecast_probabilities,
			)
		)
