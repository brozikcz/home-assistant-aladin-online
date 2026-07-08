from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from homeassistant.const import (
	PERCENTAGE,
	UnitOfLength,
	UnitOfPressure,
	UnitOfSpeed,
	UnitOfTemperature,
	UnitOfVolumetricFlux,
)
from homeassistant.components.sensor import (
	SensorDeviceClass,
	SensorEntity as ComponentSensorEntity,
	SensorEntityDescription as ComponentSensorEntityDescription,
	SensorStateClass,
)
from homeassistant.const import (
	CONF_NAME,
)
from homeassistant.core import callback, HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator
from homeassistant.helpers.device_registry import DeviceEntryType
from types import MappingProxyType
from typing import Dict
from . import AladinOnlineConfigEntry
from .aladin_online import AladinActualWeather
from .radar_coordinator import AladinRadarCoordinator
from .const import (
	DOMAIN,
	NAME,
)


class SensorType(StrEnum):
	APPARENT_TEMPERATURE = "apparent_temperature"
	CLOUDS = "clouds"
	HUMIDITY = "humidity"
	PRECIPITATION = "precipitation"
	PRESSURE = "pressure"
	SNOW_PRECIPITATION = "snow_precipitation"
	TEMPERATURE = "temperature"
	WIND_SPEED = "wind_speed"
	WIND_GUST_SPEED = "wind_gust_speed"
	RADAR_NEAREST_RAIN_DISTANCE = "radar_nearest_rain_distance"
	RADAR_MINUTES_UNTIL_RAIN = "radar_minutes_until_rain"
	RADAR_RAIN_INTENSITY = "radar_rain_intensity"
	RADAR_RAIN_PROBABILITY = "radar_rain_probability"

@dataclass(frozen=True, kw_only=True)
class SensorEntityDescription(ComponentSensorEntityDescription):
	value_func: Callable | None = None

RADAR_SENSORS: Dict[SensorType, SensorEntityDescription] = {
	SensorType.RADAR_NEAREST_RAIN_DISTANCE: SensorEntityDescription(
		key=SensorType.RADAR_NEAREST_RAIN_DISTANCE,
		name="Radar nearest rain distance",
		icon="mdi:radar",
		native_unit_of_measurement=UnitOfLength.KILOMETERS,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda data: data.nearest_distance,
	),
	SensorType.RADAR_MINUTES_UNTIL_RAIN: SensorEntityDescription(
		key=SensorType.RADAR_MINUTES_UNTIL_RAIN,
		name="Radar minutes until rain",
		icon="mdi:clock-outline",
		native_unit_of_measurement="min",
		suggested_display_precision=0,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda data: data.minutes_until_rain,
	),
	SensorType.RADAR_RAIN_INTENSITY: SensorEntityDescription(
		key=SensorType.RADAR_RAIN_INTENSITY,
		name="Radar rain intensity",
		device_class=SensorDeviceClass.PRECIPITATION_INTENSITY,
		native_unit_of_measurement=UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR,
		suggested_display_precision=2,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda data: data.rain_now_value,
	),
	SensorType.RADAR_RAIN_PROBABILITY: SensorEntityDescription(
		key=SensorType.RADAR_RAIN_PROBABILITY,
		name="Radar rain probability",
		icon="mdi:water-percent",
		native_unit_of_measurement=PERCENTAGE,
		suggested_display_precision=0,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda data: data.rain_probability,
	),
}

SENSORS: Dict[SensorType, SensorEntityDescription] = {
	SensorType.APPARENT_TEMPERATURE: SensorEntityDescription(
		key=SensorType.APPARENT_TEMPERATURE,
		device_class=SensorDeviceClass.TEMPERATURE,
		native_unit_of_measurement=UnitOfTemperature.CELSIUS,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.apparent_temperature,
	),
	SensorType.CLOUDS: SensorEntityDescription(
		key=SensorType.CLOUDS,
		icon="mdi:weather-partly-cloudy",
		native_unit_of_measurement=PERCENTAGE,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.clouds,
	),
	SensorType.HUMIDITY: SensorEntityDescription(
		key=SensorType.HUMIDITY,
		device_class=SensorDeviceClass.HUMIDITY,
		native_unit_of_measurement=PERCENTAGE,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.humidity,
	),
	SensorType.PRECIPITATION: SensorEntityDescription(
		key=SensorType.PRECIPITATION,
		device_class=SensorDeviceClass.PRECIPITATION_INTENSITY,
		native_unit_of_measurement=UnitOfVolumetricFlux.MILLIMETERS_PER_HOUR,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.precipitation,
	),
	SensorType.PRESSURE: SensorEntityDescription(
		key=SensorType.PRESSURE,
		device_class=SensorDeviceClass.ATMOSPHERIC_PRESSURE,
		native_unit_of_measurement=UnitOfPressure.HPA,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.pressure,
	),
	SensorType.SNOW_PRECIPITATION: SensorEntityDescription(
		key=SensorType.SNOW_PRECIPITATION,
		icon="mdi:weather-snowy",
		native_unit_of_measurement=PERCENTAGE,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.snow_precipitation,
	),
	SensorType.TEMPERATURE: SensorEntityDescription(
		key=SensorType.TEMPERATURE,
		device_class=SensorDeviceClass.TEMPERATURE,
		native_unit_of_measurement=UnitOfTemperature.CELSIUS,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.temperature,
	),
	SensorType.WIND_SPEED: SensorEntityDescription(
		key=SensorType.WIND_SPEED,
		device_class=SensorDeviceClass.WIND_SPEED,
		native_unit_of_measurement=UnitOfSpeed.METERS_PER_SECOND,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.wind_speed,
	),
	SensorType.WIND_GUST_SPEED: SensorEntityDescription(
		key=SensorType.WIND_GUST_SPEED,
		device_class=SensorDeviceClass.WIND_SPEED,
		native_unit_of_measurement=UnitOfSpeed.METERS_PER_SECOND,
		suggested_display_precision=1,
		state_class=SensorStateClass.MEASUREMENT,
		value_func=lambda actual_weather: actual_weather.wind_gust_speed,
	),
}


async def async_setup_entry(hass: HomeAssistant, config_entry: AladinOnlineConfigEntry, async_add_entities) -> None:
	coordinator = config_entry.runtime_data

	for sensor_type in SENSORS:
		async_add_entities([
			SensorEntity(coordinator, config_entry.data, SENSORS[sensor_type]),
		])

	for sensor_type in RADAR_SENSORS:
		async_add_entities([
			RadarSensorEntity(coordinator, config_entry.data, RADAR_SENSORS[sensor_type]),
		])


class SensorEntity(CoordinatorEntity, ComponentSensorEntity):

	entity_description: SensorEntityDescription

	_attr_has_entity_name = True

	def __init__(self, coordinator: DataUpdateCoordinator, config: MappingProxyType, entity_description: SensorEntityDescription):
		super().__init__(coordinator)

		self.entity_description = entity_description
		self._attr_translation_key = entity_description.key

		self._attr_unique_id = "{}.{}".format(
			config[CONF_NAME],
			self.entity_description.key,
		)

		self._attr_device_info = DeviceInfo(
			identifiers={(DOMAIN, config[CONF_NAME])},
			model="Weather forecast",
			name=config[CONF_NAME],
			manufacturer=NAME,
			entry_type=DeviceEntryType.SERVICE,
		)

		self._update_attributes()

	def _update_attributes(self):
		if self.coordinator.data is None or self.coordinator.data.weather is None:
			return

		actual_weather: AladinActualWeather = self.coordinator.data.weather.actual_weather

		self._attr_native_value = self.entity_description.value_func(actual_weather)


class RadarSensorEntity(CoordinatorEntity, ComponentSensorEntity):

	entity_description: SensorEntityDescription

	_attr_has_entity_name = True

	def __init__(self, coordinator: DataUpdateCoordinator, config: MappingProxyType, entity_description: SensorEntityDescription):
		super().__init__(coordinator)

		self.entity_description = entity_description
		self._attr_translation_key = entity_description.key

		# Set name explicitly to ensure descriptive entity ID
		self._attr_name = entity_description.name

		self._attr_unique_id = "{}.{}".format(
			config[CONF_NAME],
			self.entity_description.key,
		)

		self._attr_device_info = DeviceInfo(
			identifiers={(DOMAIN, f"{config[CONF_NAME]}_radar")},
			name=config[CONF_NAME],
			manufacturer="ČHMÚ",
			model="Nowcasting Engine (INCA-CZ / COTREC)",
			entry_type=DeviceEntryType.SERVICE,
			via_device=(DOMAIN, config[CONF_NAME]),
		)

		self._update_attributes()

	def _update_attributes(self):
		if self.coordinator.data is None or self.coordinator.data.radar is None:
			return

		self._attr_native_value = self.entity_description.value_func(self.coordinator.data.radar)

		if self.entity_description.key == SensorType.RADAR_RAIN_PROBABILITY:
			self._attr_extra_state_attributes = self.coordinator.data.radar.forecast_probabilities

	@callback
	def _handle_coordinator_update(self) -> None:
		self._update_attributes()
		super()._handle_coordinator_update()
