import logging
from typing import Final

LOGGER: Final = logging.getLogger(__package__)
DOMAIN: Final = "aladin_online_chmi"
NAME: Final = "Aladin online (Czech Republic) CHMI"
URL: Final = "https://data-provider.chmi.cz/api/graphs/graf.meteogram/?x={}&y={}"

CONF_RADAR_RADIUS: Final = "radar_radius"
DEFAULT_RADAR_RADIUS: Final = 10

CONF_RADAR_THRESHOLD_MMH: Final = "threshold_mmh"
DEFAULT_RADAR_THRESHOLD_MMH: Final = 0.5

CONF_RADAR_WINDOW_SIZE: Final = "window_size"
DEFAULT_RADAR_WINDOW_SIZE: Final = 3

CONF_RADAR_SIZE_THRESHOLD: Final = "size_threshold"
DEFAULT_RADAR_SIZE_THRESHOLD: Final = 2

CONF_RADAR_IMAGE_TYPE: Final = "radar_image_type"
RADAR_IMAGE_TYPE_MAX3D: Final = "max3d"
RADAR_IMAGE_TYPE_CAPPI: Final = "cappi"
DEFAULT_RADAR_IMAGE_TYPE: Final = RADAR_IMAGE_TYPE_CAPPI

CONF_WEATHER_ENTITY: Final = "weather_entity"
DEFAULT_WEATHER_ENTITY: Final = None

CONF_USE_3D_WIND_PROFILE: Final = "use_3d_wind_profile"
DEFAULT_USE_3D_WIND_PROFILE: Final = False
