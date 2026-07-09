from __future__ import annotations
import math
import io
from typing import Any, Final
from PIL import Image

# CHMI Radar Bounding Box (EPSG:3857 approximations for the PNG)
LAT_MIN: Final = 48.047
LAT_MAX: Final = 52.167
LON_MIN: Final = 11.267
LON_MAX: Final = 20.770

# Color mapping from prsi.py
COLOR_TO_DBZ = {
    (56, 0, 112): 4, (48, 0, 168): 8,
    (1, 1, 246): 12, (0, 108, 192): 16,
    (0, 160, 0): 20, (0, 188, 0): 24,
    (52, 216, 0): 28, (152, 215, 1): 32,
    (219, 215, 1): 36, (252, 176, 0): 40,
    (252, 132, 0): 44, (252, 88, 0): 48,
    (252, 0, 0): 52, (153, 2, 2): 56,
    (252, 252, 252): 60
}

DBZ_TO_MMH: Final = {
    4: 0.1, 8: 0.325, 12: 0.55, 16: 0.775,
    20: 1.0, 24: 3.25, 28: 5.5, 32: 7.75,
    36: 10.0, 40: 28.0, 44: 46.0, 48: 64.0,
    52: 82.0, 56: 100.0, 60: 100.0
}


def gps_to_pixel(lat: float, lon: float, width: int, height: int) -> tuple[int, int] | None:
    """Convert GPS coordinates to pixel coordinates using the full image canvas."""
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None

    def mercator_y(lat_deg: float) -> float:
        lat_rad = math.radians(lat_deg)
        return math.log(math.tan(math.pi / 4 + lat_rad / 2))

    y_min_merc = mercator_y(LAT_MIN)
    y_max_merc = mercator_y(LAT_MAX)
    y_target_merc = mercator_y(lat)

    x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * width)
    y = int((y_max_merc - y_target_merc) / (y_max_merc - y_min_merc) * height)

    if 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def get_dbz(r: int, g: int, b: int) -> int:
    """Find the closest dBZ value for a given RGB color. Returns 0 if it's a UI artifact."""
    dbz = 0
    min_distance = float("inf")
    for (cr, cg, cb), value in COLOR_TO_DBZ.items():
        distance = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if distance < min_distance:
            min_distance = distance
            dbz = value

    # Safety threshold against black lines, text, and gray masks snapping to white (60 dBZ)
    if min_distance > 1500:
        return 0

    return dbz


def is_precipitation_pixel(pixel: tuple[int, ...]) -> bool:
    """Determine if a pixel represents active precipitation based on alpha and color thresholds."""
    if len(pixel) == 4:
        r, g, b, alpha = pixel
        return alpha > 0 and (r > 0 or g > 0 or b > 0)
    return any(c > 0 for c in pixel[:3])


def is_significant_rain(pixel: tuple[int, ...], threshold_mmh: float = 0.5) -> bool:
    """Check if a pixel represents precipitation stronger than a defined threshold."""
    if not is_precipitation_pixel(pixel):
        return False
    r, g, b = pixel[:3]
    dbz = get_dbz(r, g, b)
    intensity = DBZ_TO_MMH.get(dbz, 0.0)
    return intensity >= threshold_mmh


def pixel_to_gps(x: int, y: int, width: int, height: int) -> tuple[float, float] | None:
    """Reverse calculation from pixel to GPS coordinates using the full canvas."""
    if not (0 <= x < width and 0 <= y < height):
        return None

    lon = LON_MIN + (x / width) * (LON_MAX - LON_MIN)

    def mercator_y(lat_deg: float) -> float:
        lat_rad = math.radians(lat_deg)
        return math.log(math.tan(math.pi / 4 + lat_rad / 2))

    y_min_merc = mercator_y(LAT_MIN)
    y_max_merc = mercator_y(LAT_MAX)

    y_target_merc = y_max_merc - (y / height) * (y_max_merc - y_min_merc)
    lat_rad = 2 * (math.atan(math.exp(y_target_merc)) - math.pi / 4)
    lat = math.degrees(lat_rad)

    return round(lat, 6), round(lon, 6)


def get_radar_info(image_bytes: bytes, lat: float, lon: float, radius: int = 60, threshold_mmh: float = 0.5) -> dict[
    str, Any]:
    """Process the current radar image to find immediate rain data and scan for the nearest precipitation."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGBA")
        width, height = img.size
        pixel_coords = gps_to_pixel(lat, lon, width, height)

        if not pixel_coords:
            return {"rain_now": False, "rain_now_value": 0.0, "nearest_distance": None}

        px, py = pixel_coords
        pixels = img.load()
        pixel_val = pixels[px, py]

        rain_now = is_significant_rain(pixel_val, threshold_mmh)
        rain_now_value = 0.0
        if rain_now:
            r, g, b = pixel_val[:3]
            dbz = get_dbz(r, g, b)
            rain_now_value = DBZ_TO_MMH.get(dbz, 0.0)

        nearest_distance = None
        nearest_pixel = None

        if not rain_now:
            for r in range(1, radius + 1):
                best_dist = float('inf')
                best_px = None

                for i in range(-r, r + 1):
                    for dx, dy in [(i, -r), (i, r)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if is_significant_rain(pixels[nx, ny], threshold_mmh):
                                d = math.sqrt(dx * dx + dy * dy)
                                if d < best_dist:
                                    best_dist = d
                                    best_px = (nx, ny)

                    if -r < i < r:
                        for dx, dy in [(-r, i), (r, i)]:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if is_significant_rain(pixels[nx, ny], threshold_mmh):
                                    d = math.sqrt(dx * dx + dy * dy)
                                    if d < best_dist:
                                        best_dist = d
                                        best_px = (nx, ny)

                if best_px is not None:
                    nearest_distance = best_dist
                    nearest_pixel = best_px
                    break

        nearest_gps = None
        if nearest_pixel:
            nearest_gps = pixel_to_gps(nearest_pixel[0], nearest_pixel[1], width, height)

        return {
            "rain_now": rain_now,
            "rain_now_value": rain_now_value,
            "nearest_distance": round(nearest_distance, 1) if nearest_distance is not None else None,
            "nearest_pixel": nearest_pixel,
            "nearest_gps": nearest_gps
        }


def check_forecast_rain(image_bytes: bytes, lat: float, lon: float, threshold_mmh: float = 0.5) -> bool:
    """Check a 3x3 km neighborhood in the forecast image for the presence of significant rain."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGBA")
        width, height = img.size
        pixel_coords = gps_to_pixel(lat, lon, width, height)

        if not pixel_coords:
            return False

        px, py = pixel_coords
        pixels = img.load()

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if is_significant_rain(pixels[nx, ny], threshold_mmh):
                        return True
        return False


def calculate_forecast_probability(image_bytes: bytes, lat: float, lon: float, window_radius: int = 3,
                                   threshold_mmh: float = 0.0) -> int:
    """Calculate spatial rain probability (%) within a bounded window around the target coordinates."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGBA")
        width, height = img.size
        pixel_coords = gps_to_pixel(lat, lon, width, height)

        if not pixel_coords:
            return 0

        px, py = pixel_coords
        pixels = img.load()

        rain_pixels = 0
        total_valid_pixels = 0

        for dx in range(-window_radius, window_radius + 1):
            for dy in range(-window_radius, window_radius + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height:
                    total_valid_pixels += 1
                    if threshold_mmh > 0.0:
                        if is_significant_rain(pixels[nx, ny], threshold_mmh):
                            rain_pixels += 1
                    else:
                        # For probability, we catch even light rain unless threshold is strictly set
                        if is_precipitation_pixel(pixels[nx, ny]) and get_dbz(pixels[nx, ny][0], pixels[nx, ny][1],
                                                                              pixels[nx, ny][2]) > 0:
                            rain_pixels += 1

        if total_valid_pixels == 0:
            return 0

        return int((rain_pixels / total_valid_pixels) * 100)