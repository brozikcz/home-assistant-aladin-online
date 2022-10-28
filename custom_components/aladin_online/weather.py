import datetime
from homeassistant import config_entries, core
from homeassistant.const import (
	CONF_NAME,
	UnitOfLength,
	UnitOfPressure,
	UnitOfSpeed,
	UnitOfTemperature,
)
from homeassistant.components.weather import (
	ATTR_FORECAST_CONDITION,
	ATTR_FORECAST_NATIVE_TEMP,
	ATTR_FORECAST_NATIVE_PRECIPITATION,
	ATTR_FORECAST_NATIVE_PRESSURE,
	ATTR_FORECAST_NATIVE_WIND_SPEED,
	ATTR_FORECAST_TIME,
	ATTR_FORECAST_WIND_BEARING,
	WeatherEntity as ComponentWeatherEntity,
)
from homeassistant.core import callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator
from homeassistant.helpers.device_registry import DeviceEntryType
from types import MappingProxyType
from .aladin_online import AladinActualWeather
from .const import (
	DOMAIN,
	DATA_COORDINATOR,
	NAME,
)

async def async_setup_entry(hass: core.HomeAssistant, config_entry: config_entries.ConfigEntry, async_add_entities) -> None:
	coordinator = hass.data[DOMAIN][config_entry.entry_id][DATA_COORDINATOR]

	async_add_entities([
		WeatherEntity(coordinator, config_entry.data),
	])


class WeatherEntity(CoordinatorEntity, ComponentWeatherEntity):

	_attr_has_entity_name = True
	_attr_native_precipitation_unit = UnitOfLength.MILLIMETERS
	_attr_native_pressure_unit = UnitOfPressure.HPA
	_attr_native_temperature_unit = UnitOfTemperature.CELSIUS
	_attr_native_wind_speed_unit = UnitOfSpeed.METERS_PER_SECOND

	def __init__(self, coordinator: DataUpdateCoordinator, config: MappingProxyType):
		super().__init__(coordinator)

		self._attr_unique_id = "{}.{}".format(
			config[CONF_NAME],
			"hourly",
		)

		self._attr_device_info = DeviceInfo(
			identifiers={(DOMAIN,)},
			model="Weather forecast",
			name=config[CONF_NAME],
			manufacturer=NAME,
			entry_type=DeviceEntryType.SERVICE,
		)

		self._update_attributes()

	def _update_attributes(self):
		if self.coordinator.data is None:
			return

		actual_weather: AladinActualWeather = self.coordinator.data.actual_weather

		self._attr_condition = actual_weather.condition
		self._attr_humidity = actual_weather.humidity
		self._attr_native_pressure = actual_weather.pressure
		self._attr_native_temperature = actual_weather.temperature
		self._attr_native_wind_speed = actual_weather.wind_speed
		self._attr_wind_bearing = actual_weather.wind_bearing

		now = datetime.datetime.now()

		self._attr_forecast = []

		for hourly_forecast in self.coordinator.data.hourly_forecasts:
			if hourly_forecast.datetime < now:
				continue

			self._attr_forecast.append({
				ATTR_FORECAST_TIME: hourly_forecast.datetime,
				ATTR_FORECAST_CONDITION: hourly_forecast.condition,
				ATTR_FORECAST_NATIVE_TEMP: hourly_forecast.temperature,
				ATTR_FORECAST_NATIVE_PRECIPITATION: hourly_forecast.precipitation,
				ATTR_FORECAST_NATIVE_PRESSURE: hourly_forecast.pressure,
				ATTR_FORECAST_NATIVE_WIND_SPEED: hourly_forecast.wind_speed,
				ATTR_FORECAST_WIND_BEARING: hourly_forecast.wind_bearing,
			})

	@callback
	def _handle_coordinator_update(self) -> None:
		self._update_attributes()
		super()._handle_coordinator_update()
