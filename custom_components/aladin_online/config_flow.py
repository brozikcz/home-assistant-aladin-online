from __future__ import annotations
from homeassistant import config_entries
from homeassistant.const import (
	CONF_NAME,
	CONF_LATITUDE,
	CONF_LONGITUDE,
)
from homeassistant.data_entry_flow import AbortFlow, FlowResult
from homeassistant.core import callback
from homeassistant.helpers import aiohttp_client, selector
from http import HTTPStatus
import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from .const import (
	DOMAIN, NAME, URL, LOGGER, CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS,
	CONF_RADAR_THRESHOLD_MMH, DEFAULT_RADAR_THRESHOLD_MMH,
	CONF_RADAR_WINDOW_SIZE, DEFAULT_RADAR_WINDOW_SIZE,
	CONF_RADAR_SIZE_THRESHOLD, DEFAULT_RADAR_SIZE_THRESHOLD,
	CONF_RADAR_IMAGE_TYPE, RADAR_IMAGE_TYPE_MAX3D, RADAR_IMAGE_TYPE_CAPPI, DEFAULT_RADAR_IMAGE_TYPE,
)
from .errors import LocationUnavailable, ServiceUnavailable
from typing import Any, Dict

class AladinOnlineConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
	"""Weather forecast config flow."""

	async def async_step_user(self, user_input: Dict[str, Any] | None = None) -> FlowResult:
		errors = {}
		if user_input is not None:
			try:
				config = {
					CONF_NAME: user_input[CONF_NAME],
					CONF_LATITUDE: user_input[CONF_LATITUDE],
					CONF_LONGITUDE: user_input[CONF_LONGITUDE],
				}
				await self.async_set_unique_id(user_input[CONF_NAME])
				self._abort_if_unique_id_configured()

				await self._async_validate_location(user_input[CONF_LATITUDE], user_input[CONF_LONGITUDE])

				return self.async_create_entry(title=NAME, data=config)
			except AbortFlow as ex:
				return self.async_abort(reason=ex.reason)
			except ServiceUnavailable:
				errors["base"] = "service_unavailable"
			except LocationUnavailable:
				errors["base"] = "location_unavailable"
			except Exception:
				LOGGER.error("Unknown error connecting to %s", URL.format(user_input[CONF_LONGITUDE], user_input[CONF_LATITUDE]))
				return self.async_abort(reason="unknown")

		return self.async_show_form(
			step_id="user",
			data_schema=vol.Schema({
				vol.Required(CONF_NAME, default=self.hass.config.location_name): str,
				vol.Required(CONF_LATITUDE, default=self.hass.config.latitude): cv.latitude,
				vol.Required(CONF_LONGITUDE, default=self.hass.config.longitude): cv.longitude,
			}),
			errors=errors,
		)

	async def async_step_reconfigure(self, user_input: Dict[str, Any] | None = None) -> FlowResult:
		errors = {}
		reconfigure_entry = self._get_reconfigure_entry()
		if user_input is not None:
			try:
				config = {
					CONF_NAME: reconfigure_entry.data[CONF_NAME],
					CONF_LATITUDE: user_input[CONF_LATITUDE],
					CONF_LONGITUDE: user_input[CONF_LONGITUDE],
				}

				await self._async_validate_location(user_input[CONF_LATITUDE], user_input[CONF_LONGITUDE])

				return self.async_update_reload_and_abort(reconfigure_entry, data=config)
			except AbortFlow as ex:
				return self.async_abort(reason=ex.reason)
			except ServiceUnavailable:
				errors["base"] = "service_unavailable"
			except LocationUnavailable:
				errors["base"] = "location_unavailable"
			except Exception:
				LOGGER.error("Unknown error connecting to %s", URL.format(user_input[CONF_LONGITUDE], user_input[CONF_LATITUDE]))
				return self.async_abort(reason="unknown")

		return self.async_show_form(
			step_id="reconfigure",
			data_schema=vol.Schema({
				vol.Required(CONF_LATITUDE, default=reconfigure_entry.data.get(CONF_LATITUDE, self.hass.config.latitude)): cv.latitude,
				vol.Required(CONF_LONGITUDE, default=reconfigure_entry.data.get(CONF_LONGITUDE, self.hass.config.longitude)): cv.longitude,
			}),
			errors=errors,
		)

	async def _async_validate_location(self, latitude: float, longitude: float) -> None:
		session = aiohttp_client.async_get_clientsession(self.hass)
		response = await session.get(URL.format(longitude, latitude))
		if response.status != HTTPStatus.OK:
			raise ServiceUnavailable
		if await response.text() == "":
			raise LocationUnavailable

	@staticmethod
	@callback
	def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> AladinOnlineOptionsFlowHandler:
		"""Get the options flow for this handler."""
		return AladinOnlineOptionsFlowHandler()


class AladinOnlineOptionsFlowHandler(config_entries.OptionsFlowWithReload):
	"""Handle Aladin Online options."""

	async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
		"""Manage the options."""
		if user_input is not None:
			return self.async_create_entry(title="", data=user_input)

		# Get current window size to determine max size_threshold
		current_window_size = int(self.config_entry.options.get(CONF_RADAR_WINDOW_SIZE, DEFAULT_RADAR_WINDOW_SIZE))
		max_size_threshold = current_window_size * current_window_size

		return self.async_show_form(
			step_id="init",
			data_schema=vol.Schema({
				vol.Optional(
					CONF_RADAR_RADIUS,
					default=int(self.config_entry.options.get(CONF_RADAR_RADIUS, self.config_entry.data.get(CONF_RADAR_RADIUS, DEFAULT_RADAR_RADIUS))),
				): selector.NumberSelector(
					selector.NumberSelectorConfig(
						min=1,
						max=100,
						mode=selector.NumberSelectorMode.BOX,
					)
				),
				vol.Optional(
					CONF_RADAR_THRESHOLD_MMH,
					default=str(self.config_entry.options.get(CONF_RADAR_THRESHOLD_MMH, DEFAULT_RADAR_THRESHOLD_MMH)),
				): selector.SelectSelector(
					selector.SelectSelectorConfig(
						options=[
							selector.SelectOptionDict(value="0.1", label="0.1 mm/h"),
							selector.SelectOptionDict(value="0.3", label="0.3 mm/h"),
							selector.SelectOptionDict(value="0.5", label="0.5 mm/h"),
							selector.SelectOptionDict(value="1.0", label="1.0 mm/h"),
							selector.SelectOptionDict(value="2.5", label="2.5 mm/h"),
						],
						mode=selector.SelectSelectorMode.DROPDOWN,
					)
				),
				vol.Optional(
					CONF_RADAR_WINDOW_SIZE,
					default=str(self.config_entry.options.get(CONF_RADAR_WINDOW_SIZE, DEFAULT_RADAR_WINDOW_SIZE)),
				): selector.SelectSelector(
					selector.SelectSelectorConfig(
						options=[
							selector.SelectOptionDict(value="1", label="1x1 km"),
							selector.SelectOptionDict(value="3", label="3x3 km"),
							selector.SelectOptionDict(value="5", label="5x5 km"),
						],
						mode=selector.SelectSelectorMode.DROPDOWN,
					)
				),
				vol.Optional(
					CONF_RADAR_SIZE_THRESHOLD,
					default=int(self.config_entry.options.get(CONF_RADAR_SIZE_THRESHOLD, DEFAULT_RADAR_SIZE_THRESHOLD)),
				): selector.NumberSelector(
					selector.NumberSelectorConfig(
						min=1,
						max=max_size_threshold,
						mode=selector.NumberSelectorMode.BOX,
					)
				),
				vol.Optional(
					CONF_RADAR_IMAGE_TYPE,
					default=str(self.config_entry.options.get(CONF_RADAR_IMAGE_TYPE, DEFAULT_RADAR_IMAGE_TYPE)),
				): selector.SelectSelector(
					selector.SelectSelectorConfig(
						options=[
							selector.SelectOptionDict(value=RADAR_IMAGE_TYPE_MAX3D, label="MAX_Z_mask"),
							selector.SelectOptionDict(value=RADAR_IMAGE_TYPE_CAPPI, label="PseudoCAPPI 2km"),
						],
						mode=selector.SelectSelectorMode.DROPDOWN,
					)
				),
			}),
		)
