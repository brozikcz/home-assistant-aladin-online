from __future__ import annotations
from homeassistant.components.binary_sensor import (
	BinarySensorEntity,
)
from homeassistant.const import CONF_NAME
from homeassistant.core import callback, HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.helpers.device_registry import DeviceEntryType
from types import MappingProxyType
from . import AladinOnlineConfigEntry
from .radar_coordinator import AladinRadarCoordinator
from .const import (
	DOMAIN,
	NAME,
)

async def async_setup_entry(hass: HomeAssistant, config_entry: AladinOnlineConfigEntry, async_add_entities) -> None:
	coordinator = config_entry.runtime_data
	async_add_entities([
		RadarRainNowBinarySensor(coordinator, config_entry.data),
	])

class RadarRainNowBinarySensor(CoordinatorEntity[AladinRadarCoordinator], BinarySensorEntity):
	_attr_has_entity_name = True
	_attr_name = "Radar rain now"
	_attr_translation_key = "radar_rain_now"

	def __init__(self, coordinator: AladinRadarCoordinator, config: MappingProxyType):
		super().__init__(coordinator)
		self._attr_unique_id = "{}.radar_rain_now".format(config[CONF_NAME])
		self._attr_device_info = DeviceInfo(
			identifiers={(DOMAIN, f"{config[CONF_NAME]}_radar")},
			name=config[CONF_NAME],
			manufacturer="ČHMÚ",
			model="Nowcasting Engine (INCA-CZ / COTREC)",
			entry_type=DeviceEntryType.SERVICE,
			via_device=(DOMAIN, config[CONF_NAME]),
		)

	@property
	def is_on(self) -> bool | None:
		if self.coordinator.data is None or self.coordinator.data.radar is None:
			return None
		return self.coordinator.data.radar.rain_now
