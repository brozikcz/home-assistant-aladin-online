from homeassistant import config_entries, core
from homeassistant.const import Platform
from .aladin_online import AladinOnlineCoordinator
from .const import DATA_COORDINATOR, DOMAIN


async def async_setup_entry(hass: core.HomeAssistant, config_entry: config_entries.ConfigEntry) -> bool:
	hass.data.setdefault(DOMAIN, {})
	hass.data[DOMAIN][config_entry.entry_id] = {}

	coordinator = AladinOnlineCoordinator(hass, config_entry.data)

	await coordinator.async_refresh()

	hass.data[DOMAIN][config_entry.entry_id][DATA_COORDINATOR] = coordinator

	for platform in (Platform.SENSOR, Platform.WEATHER):
		hass.async_create_task(
			hass.config_entries.async_forward_entry_setup(config_entry, platform)
		)

	return True
