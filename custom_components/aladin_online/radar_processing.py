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
 
# dBZ to mm/h mapping based on Marshall-Palmer relation
DBZ_TO_MMH: Final = {
	4: 0.1, 8: 0.325, 12: 0.55, 16: 0.775,
	20: 1.0, 24: 3.25, 28: 5.5, 32: 7.75,
	36: 10.0, 40: 28.0, 44: 46.0, 48: 64.0,
	52: 82.0, 56: 100.0, 60: 100.0
}

def gps_to_pixel(lat: float, lon: float, width: int, height: int) -> tuple[int, int] | None:
	"""Convert GPS coordinates to pixel coordinates for the CHMI radar image."""
	if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
		return None

	# Helper function for Mercator Y projection
	def mercator_y(lat_deg: float) -> float:
		lat_rad = math.radians(lat_deg)
		return math.log(math.tan(math.pi / 4 + lat_rad / 2))

	y_min_merc = mercator_y(LAT_MIN)
	y_max_merc = mercator_y(LAT_MAX)
	y_target_merc = mercator_y(lat)

	# X remains linear
	x = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * width)
	
	# Y is interpolated using the Mercator scale (remember image Y=0 is top)
	y = int((y_max_merc - y_target_merc) / (y_max_merc - y_min_merc) * height)

	if 0 <= x < width and 0 <= y < height:
		return x, y
	return None

def get_dbz(r: int, g: int, b: int) -> int:
	"""Find the closest dBZ value for a given RGB color."""
	dbz = 0
	min_distance = float("inf")
	for (cr, cg, cb), value in COLOR_TO_DBZ.items():
		distance = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
		if distance < min_distance:
			min_distance = distance
			dbz = value
	return dbz

def is_precipitation_pixel(pixel: tuple[int, ...]) -> bool:
	"""Determine if a pixel represents precipitation based on alpha and color."""
	if len(pixel) == 4:  # RGBA
		r, g, b, alpha = pixel
		# In prsi.py: alpha > 0 and (r | g | b) != 0
		return alpha > 0 and (r > 0 or g > 0 or b > 0)
	
	return any(c > 0 for c in pixel[:3])

def get_radar_info(image_bytes: bytes, lat: float, lon: float, radius: int = 60) -> dict[str, Any]:
	"""Process radar image to find rain at location and nearest rain."""
	with Image.open(io.BytesIO(image_bytes)) as img:
		img = img.convert("RGBA")
		width, height = img.size
		pixel_coords = gps_to_pixel(lat, lon, width, height)
		
		if not pixel_coords:
			return {"rain_now": False, "nearest_distance": None}

		px, py = pixel_coords
		pixels = img.load()
		pixel_val = pixels[px, py]

		# 1. Rain exactly at coordinates
		rain_now = is_precipitation_pixel(pixel_val)
		rain_now_value = 0.0
		if rain_now:
			r, g, b = pixel_val[:3]
			dbz = get_dbz(r, g, b)
			rain_now_value = DBZ_TO_MMH.get(dbz, 0.0)
		
		# 2. Nearest rain distance (spiral search up to radius km)
		nearest_distance = 0 if rain_now else None
		found = False
		if not rain_now:
			for r in range(1, radius + 1):
				for i in range(-r, r + 1):
					# Top and bottom edges
					for dx, dy in [(i, -r), (i, r)]:
						nx, ny = px + dx, py + dy
						if 0 <= nx < width and 0 <= ny < height:
							if is_precipitation_pixel(pixels[nx, ny]):
								nearest_distance = math.sqrt(dx*dx + dy*dy)
								found = True
								break
					if found: break
					
					# Left and right edges
					for dx, dy in [(-r, i), (r, i)]:
						nx, ny = px + dx, py + dy
						if 0 <= nx < width and 0 <= ny < height:
							if is_precipitation_pixel(pixels[nx, ny]):
								nearest_distance = math.sqrt(dx*dx + dy*dy)
								found = True
								break
					if found: break
				if found: break

		return {
			"rain_now": rain_now,
			"rain_now_value": rain_now_value,
			"nearest_distance": round(nearest_distance, 1) if nearest_distance is not None else None
		}

def check_forecast_rain(image_bytes: bytes, lat: float, lon: float) -> bool:
	"""Check if there is rain at the location in a forecast image."""
	with Image.open(io.BytesIO(image_bytes)) as img:
		img = img.convert("RGBA")
		width, height = img.size
		pixel_coords = gps_to_pixel(lat, lon, width, height)
		
		if not pixel_coords:
			return False

		px, py = pixel_coords
		pixels = img.load()
		return is_precipitation_pixel(pixels[px, py])

def calculate_forecast_probability(image_bytes: bytes, lat: float, lon: float, window_radius: int = 3) -> int:
	"""Calculate rain probability in a window around the coordinates."""
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
					if is_precipitation_pixel(pixels[nx, ny]):
						rain_pixels += 1
		
		if total_valid_pixels == 0:
			return 0
			
		return int((rain_pixels / total_valid_pixels) * 100)
