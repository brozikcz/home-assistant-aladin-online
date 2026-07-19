from __future__ import annotations
import math
import io
import logging
from typing import Any, Final
from PIL import Image

LOGGER = logging.getLogger(__name__)

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

# O(1) rychlé vyhledávání barev pro optimalizaci výkonu CPU
COLOR_TO_DBZ_TUPLE = {color: dbz for color, dbz in COLOR_TO_DBZ.items()}

DBZ_TO_MMH: Final = {
    4: 0.1, 8: 0.325, 12: 0.55, 16: 0.775,
    20: 1.0, 24: 3.25, 28: 5.5, 32: 7.75,
    36: 10.0, 40: 28.0, 44: 46.0, 48: 64.0,
    52: 82.0, 56: 100.0, 60: 100.0
}

EARTH_RADIUS_KM: Final = 6371.0
PSEUDOCAPPI_ALTITUDE_M: Final = 2000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points on the earth."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def get_terminal_velocity(intensity_mmh: float) -> float:
    """Returns terminal velocity of a rain drop (m/s) based on rain intensity."""
    if intensity_mmh <= 0.0:
        return 6.0
    if intensity_mmh <= 0.2:
        return 2.0
    if intensity_mmh <= 2.0:
        return 4.0
    if intensity_mmh <= 10.0:
        return 6.0
    if intensity_mmh <= 30.0:
        return 8.0
    return 9.0


def calculate_dynamic_threshold(base_threshold_mmh: float, humidity: float | None) -> float:
    """Calculate dynamic threshold based on humidity (virga compensation).
    Uses continuous linear interpolation to prevent value flapping.
    """
    if humidity is None:
        return base_threshold_mmh

    if humidity <= 50.0:
        # Vlhkost <= 50%: Práh je 4x vyšší (silné odpařování)
        multiplier = 4.0
    elif humidity <= 85.0:
        # Vlhkost 50% - 85%: Lineární pokles z 4.0 na 1.0
        # (humidity - 50) / 35 dá hodnotu od 0.0 do 1.0
        multiplier = 4.0 - ((humidity - 50.0) / 35.0) * 3.0
    elif humidity <= 95.0:
        # Vlhkost 85% - 95%: Lineární pokles z 1.0 na 0.5 (vzduch je nasycen, déšť snadno dopadne)
        multiplier = 1.0 - ((humidity - 85.0) / 10.0) * 0.5
    else:
        # Vlhkost > 95%: Práh je na polovině (detekce i slabého mrholení)
        multiplier = 0.5

    return base_threshold_mmh * multiplier


def calculate_wind_drift_offset_1d(wind_speed_ms: float, wind_bearing_deg: int, fall_time_seconds: float, lat: float) -> \
        tuple[float, float]:
    """Calculate latitude and longitude offset due to wind drift using dynamic latitude."""
    drift_distance_m = wind_speed_ms * fall_time_seconds

    if drift_distance_m < 0.01:
        return 0.0, 0.0

    wind_bearing_rad = math.radians(wind_bearing_deg)

    # Přidáno záporné znaménko pro korektní obrácení vektoru (Reverse Tracking proti směru větru)
    lat_offset_m = -drift_distance_m * math.cos(wind_bearing_rad)
    lon_offset_m = -drift_distance_m * math.sin(wind_bearing_rad)

    # Approximate conversion meters -> degrees at the specific latitude for lon
    lat_offset_deg = lat_offset_m / 111111.0
    lon_offset_deg = lon_offset_m / (111111.0 * math.cos(math.radians(lat)))

    return lat_offset_deg, lon_offset_deg


def gps_to_pixel(lat: float, lon: float, width: int, height: int) -> tuple[int, int] | None:
    """Convert GPS coordinates to pixel coordinates using the full image canvas."""
    if lat is None or lon is None:
        return None
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

    if x == width:
        x = width - 1
    if y == height:
        y = height - 1

    if 0 <= x < width and 0 <= y < height:
        return x, y
    return None


def calculate_3d_wind_drift(profile_data: list[dict], intensity_mmh: float, lat: float) -> tuple[float, float, float]:
    """
    Simulates a drop falling through multiple atmospheric layers.
    Returns: lat_offset_deg, lon_offset_deg, lowest_layer_humidity
    """
    if not profile_data:
        return 0.0, 0.0, 100.0

    v_term = get_terminal_velocity(intensity_mmh)

    total_lat_offset_m = 0.0
    total_lon_offset_m = 0.0

    # Použijeme vlhkost z nejnižší dostupné vrstvy pro efektivnější detekci virgy
    final_humidity = profile_data[-1]["humidity_pct"]

    for i in range(len(profile_data) - 1):
        top_layer = profile_data[i]
        bottom_layer = profile_data[i + 1]

        # Omezení vrstvy na fyzikální realitu (Radar v 2000m, Terén ve 293m)
        layer_top = min(top_layer["height_m"], 2000.0)
        # TODO from HA, user config flow or use open meteo `elevation` field!!
        layer_bottom = max(bottom_layer["height_m"], 293.0)

        if layer_top <= layer_bottom:
            continue

        dz = layer_top - layer_bottom

        dt = dz / v_term

        u1 = -top_layer["wind_speed_ms"] * math.sin(math.radians(top_layer["wind_dir_deg"]))
        v1 = -top_layer["wind_speed_ms"] * math.cos(math.radians(top_layer["wind_dir_deg"]))
        u2 = -bottom_layer["wind_speed_ms"] * math.sin(math.radians(bottom_layer["wind_dir_deg"]))
        v2 = -bottom_layer["wind_speed_ms"] * math.cos(math.radians(bottom_layer["wind_dir_deg"]))

        avg_u = (u1 + u2) / 2.0
        avg_v = (v1 + v2) / 2.0

        total_lon_offset_m += avg_u * dt
        total_lat_offset_m += avg_v * dt

    lat_offset_deg = total_lat_offset_m / 111111.0
    lon_offset_deg = total_lon_offset_m / (111111.0 * math.cos(math.radians(lat)))

    return lat_offset_deg, lon_offset_deg, final_humidity


def get_dbz(r: int, g: int, b: int, a: int = 255) -> int:
    """Find the closest dBZ value for a given RGB color. Returns 0 if transparent or UI artifact."""
    if a == 0:
        return 0

    # Rychlá shoda O(1) pro většinu pixelů, drasticky ulehčí CPU
    exact_match = COLOR_TO_DBZ_TUPLE.get((r, g, b))
    if exact_match is not None:
        return exact_match

    # Fallback na lineární vyhledávání (pouze pro okraje a kompresní artefakty)
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


def get_dynamic_drift_pixel(lat: float, lon: float, width: int, height: int, pixels: Any, wind_speed_ms: float,
                            wind_bearing_deg: int, profile_data: list[dict] | None = None) -> tuple[
    tuple[int, int] | None, float, float, float]:
    """
    Applies a 2-step wind drift calculation (Reverse Tracking) and returns the target pixel + applied lat/lon offsets.
    """
    if profile_data:
        lat_off_1, lon_off_1, _ = calculate_3d_wind_drift(profile_data, 0.0, lat)
        px1_coords = gps_to_pixel(lat - lat_off_1, lon - lon_off_1, width, height)

        if not px1_coords:
            return None, 0.0, 0.0, 100.0

        px1_x, px1_y = px1_coords
        dbz = get_dbz(*pixels[px1_x, px1_y])
        intensity_mmh = DBZ_TO_MMH.get(dbz, 0.0)

        lat_off_2, lon_off_2, final_humidity = calculate_3d_wind_drift(profile_data, intensity_mmh, lat)
        return gps_to_pixel(lat - lat_off_2, lon - lon_off_2, width, height), lat_off_2, lon_off_2, final_humidity

    if wind_speed_ms < 0.1:
        return gps_to_pixel(lat, lon, width, height), 0.0, 0.0, 100.0

    initial_fall_time = PSEUDOCAPPI_ALTITUDE_M / 6.0
    lat_off_1, lon_off_1 = calculate_wind_drift_offset_1d(wind_speed_ms, wind_bearing_deg, initial_fall_time, lat)
    px1_coords = gps_to_pixel(lat - lat_off_1, lon - lon_off_1, width, height)

    if not px1_coords:
        return None, 0.0, 0.0, 100.0

    px1_x, px1_y = px1_coords
    dbz = get_dbz(*pixels[px1_x, px1_y])
    intensity_mmh = DBZ_TO_MMH.get(dbz, 0.0)

    real_fall_time = (PSEUDOCAPPI_ALTITUDE_M / get_terminal_velocity(
        intensity_mmh)) if intensity_mmh > 0.0 else initial_fall_time
    lat_off_2, lon_off_2 = calculate_wind_drift_offset_1d(wind_speed_ms, wind_bearing_deg, real_fall_time, lat)

    return gps_to_pixel(lat - lat_off_2, lon - lon_off_2, width, height), lat_off_2, lon_off_2, 100.0


def evaluate_pixel_cloud(px: int, py: int, width: int, height: int, pixels: Any, window_size: int, threshold_mmh: float,
                         humidity: float | None = None) -> tuple[int, float]:
    """Scans the window_size matrix around px, py and returns (matching_pixel_count, max_intensity)."""
    dynamic_threshold = calculate_dynamic_threshold(threshold_mmh, humidity)
    half = window_size // 2
    matching_pixel_count = 0
    max_intensity = 0.0

    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            nx, ny = px + dx, py + dy
            if 0 <= nx < width and 0 <= ny < height:
                dbz = get_dbz(*pixels[nx, ny])
                intensity = DBZ_TO_MMH.get(dbz, 0.0)

                if intensity >= dynamic_threshold:
                    matching_pixel_count += 1
                    if intensity > max_intensity:
                        max_intensity = intensity

    return matching_pixel_count, max_intensity


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


def save_debug_image(img: Image.Image, original_lat: float, original_lon: float,
                     drifted_px: int, drifted_py: int, best_px: tuple[int, int] | None,
                     radius: int, width: int, height: int) -> None:
    """Draw and save a debug image highlighting the wind drift offset and detected rain."""
    from PIL import ImageDraw

    debug_img = img.copy()
    draw = ImageDraw.Draw(debug_img)

    # Define high-contrast colors
    COLOR_CITY = (0, 0, 0, 255)  # Black
    COLOR_ORIG_HOME = (128, 0, 128, 255)  # Purple
    COLOR_DRIFT_LINE = (255, 20, 147, 255)  # Pink
    COLOR_DRIFT_HOME = (255, 0, 0, 255)  # Red
    COLOR_RAIN = (0, 255, 0, 255)  # Green

    def draw_text_with_outline(x, y, text, text_color, outline_color=(255, 255, 255, 255)):
        draw.text((x - 1, y - 1), text, fill=outline_color)
        draw.text((x + 1, y - 1), text, fill=outline_color)
        draw.text((x - 1, y + 1), text, fill=outline_color)
        draw.text((x + 1, y + 1), text, fill=outline_color)
        draw.text((x, y), text, fill=text_color)

    # Reference cities
    cities = {
        "Praha": (50.0878, 14.4205),
        "Brno": (49.1951, 16.6068),
        "Ostrava": (49.8209, 18.2625),
        "Plzen": (49.7384, 13.3736),
        "Liberec": (50.7671, 15.0562),
        "C. Budejovice": (48.9745, 14.4743)
    }

    for name, (c_lat, c_lon) in cities.items():
        c_coords = gps_to_pixel(c_lat, c_lon, width, height)
        if c_coords:
            cx, cy = c_coords
            draw.rectangle([cx - 3, cy - 3, cx + 3, cy + 3], fill=COLOR_CITY)
            draw_text_with_outline(cx + 6, cy - 6, name, COLOR_CITY)

    # Original Home Location (Purple) & Wind Drift Line (Pink)
    orig_coords = gps_to_pixel(original_lat, original_lon, width, height)
    if orig_coords:
        ox, oy = orig_coords
        draw.line([(ox, oy), (drifted_px, drifted_py)], fill=COLOR_DRIFT_LINE, width=3)
        draw.rectangle([ox - 4, oy - 4, ox + 4, oy + 4], fill=COLOR_ORIG_HOME)

    # Drifted Target Location (Red) & Search Radius (Pink)
    draw.rectangle([drifted_px - radius, drifted_py - radius, drifted_px + radius, drifted_py + radius],
                   outline=COLOR_DRIFT_LINE, width=2)
    draw.rectangle([drifted_px - 4, drifted_py - 4, drifted_px + 4, drifted_py + 4], fill=COLOR_DRIFT_HOME)

    # Nearest Found Rain (Green)
    if best_px:
        nx, ny = best_px
        draw.line([(drifted_px, drifted_py), (nx, ny)], fill=COLOR_RAIN, width=3)
        draw.rectangle([nx - 4, ny - 4, nx + 4, ny + 4], fill=COLOR_RAIN)

    try:
        debug_img.save("/config/www/debug_radar.png")
    except Exception:
        try:
            debug_img.save("/tmp/debug_radar.png")
        except Exception as ex:
            LOGGER.debug("Failed to save debug image: %s", ex)


def get_radar_info(image_bytes: bytes, lat: float, lon: float, radius: int = 60, threshold_mmh: float = 0.5,
                   window_size: int = 3, size_threshold: int = 2, humidity: float | None = None,
                   wind_speed_ms: float = 0.0, wind_bearing_deg: int = 0, profile_data: list[dict] | None = None) -> \
        dict[str, Any]:
    """Process the current radar image to find immediate rain data and scan for the nearest precipitation."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGBA")
        width, height = img.size
        pixels = img.load()

        pixel_coords, lat_offset, lon_offset, final_humidity = get_dynamic_drift_pixel(
            lat, lon, width, height, pixels, wind_speed_ms, wind_bearing_deg, profile_data
        )

        if not pixel_coords:
            return {
                "rain_now": False,
                "rain_now_value": 0.0,
                "rain_now_pixel_count": 0,
                "nearest_distance": None,
            }

        px, py = pixel_coords

        # Použití přesné haversine funkce (počítáno proti směru driftu)
        pixel_shift = haversine(lat, lon, lat - lat_offset, lon - lon_offset) * 1000.0

        if profile_data:
            LOGGER.debug("Wind drift compensation (3D profile): pixel shift = %.1f m", pixel_shift)
        else:
            LOGGER.debug("Wind drift compensation (1D fallback): pixel shift = %.1f m (wind %.1f m/s, bearing %d°)",
                         pixel_shift, wind_speed_ms, wind_bearing_deg)

        effective_humidity = final_humidity if profile_data else humidity
        rain_now_pixel_count, rain_now_value = evaluate_pixel_cloud(px, py, width, height, pixels, window_size,
                                                                    threshold_mmh, effective_humidity)
        rain_now = (rain_now_pixel_count >= size_threshold)
        if not rain_now:
            rain_now_value = 0.0

        nearest_distance = None
        best_px = None

        if not rain_now:
            for r in range(1, radius + 1):
                best_dist_px = float('inf')

                for i in range(-r, r + 1):
                    for dx, dy in [(i, -r), (i, r)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            pixel_val = pixels[nx, ny]
                            if (len(pixel_val) == 4 and pixel_val[3] == 0) or (
                                    pixel_val[0] == 0 and pixel_val[1] == 0 and pixel_val[2] == 0):
                                continue

                            count, _ = evaluate_pixel_cloud(nx, ny, width, height, pixels, window_size, threshold_mmh,
                                                            effective_humidity)
                            if count >= size_threshold:
                                d = math.sqrt(dx * dx + dy * dy)
                                if d < best_dist_px:
                                    best_dist_px = d
                                    best_px = (nx, ny)

                    if -r < i < r:
                        for dx, dy in [(-r, i), (r, i)]:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                pixel_val = pixels[nx, ny]
                                if (len(pixel_val) == 4 and pixel_val[3] == 0) or (
                                        pixel_val[0] == 0 and pixel_val[1] == 0 and pixel_val[2] == 0):
                                    continue

                                count, _ = evaluate_pixel_cloud(nx, ny, width, height, pixels, window_size,
                                                                threshold_mmh, effective_humidity)
                                if count >= size_threshold:
                                    d = math.sqrt(dx * dx + dy * dy)
                                    if d < best_dist_px:
                                        best_dist_px = d
                                        best_px = (nx, ny)

                if best_px is not None:
                    # Nalezen nejbližší pixel, přepočet vzdálenosti na reálné kilometry
                    nx, ny = best_px
                    rain_gps = pixel_to_gps(nx, ny, width, height)

                    if rain_gps:
                        rain_lat, rain_lon = rain_gps
                        # Vzdálenost měříme od skutečného domova, ne od větrem posunutého bodu
                        nearest_distance = haversine(lat, lon, rain_lat, rain_lon)
                    else:
                        nearest_distance = best_dist_px  # Fallback
                    break

        # save_debug_image(img, lat, lon, px, py, best_px, radius, width, height)

        return {
            "rain_now": rain_now,
            "rain_now_value": rain_now_value,
            "rain_now_pixel_count": rain_now_pixel_count,
            "nearest_distance": round(nearest_distance, 1) if nearest_distance is not None else None,
        }


def get_forecast_info(image_bytes: bytes, lat: float, lon: float, window_size: int = 3,
                      threshold_mmh: float = 0.5, size_threshold: int = 2, humidity: float | None = None,
                      wind_speed_ms: float = 0.0, wind_bearing_deg: int = 0,
                      profile_data: list[dict] | None = None) -> dict[str, Any]:
    """Calculate comprehensive forecast info (probability, rain state, intensity, cloud size)."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGBA")
        width, height = img.size
        pixels = img.load()

        pixel_coords, _, _, final_humidity = get_dynamic_drift_pixel(
            lat, lon, width, height, pixels, wind_speed_ms, wind_bearing_deg, profile_data
        )

        if not pixel_coords:
            return {
                "rain": False,
                "probability_pct": 0,
                "intensity_mmh": 0.0,
                "cloud_size_px": 0
            }

        px, py = pixel_coords
        effective_humidity = final_humidity if profile_data else humidity

        rain_pixels, max_intensity = evaluate_pixel_cloud(px, py, width, height, pixels, window_size, threshold_mmh,
                                                          effective_humidity)

        return {
            "rain": rain_pixels >= size_threshold,
            "probability_pct": int((rain_pixels / (window_size * window_size)) * 100),
            "intensity_mmh": max_intensity if rain_pixels >= size_threshold else 0.0,
            "cloud_size_px": rain_pixels
        }