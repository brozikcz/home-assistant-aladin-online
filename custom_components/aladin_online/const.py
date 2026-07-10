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
