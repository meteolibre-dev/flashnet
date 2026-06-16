#!/usr/bin/env python3
"""
Convert forecast TIFF files into per-channel videos with proper colormaps
and European country borders overlay.

Colormaps (matching main_optimized.py):
  - lightning: yellow -> orange -> red
  - radar:     palette_radar_35 (matching main_optimized.py)
  - sat_ch1 (IR): red colormap

Produces:
  - One video per channel (forecast_lightning.mp4, forecast_radar.mp4, …)
  - A combined 2x2 panel video (forecast_combined.mp4) showing all 4
    channels side-by-side with a shared timestamp title.

Usage:
    python tiff_to_video.py <tiff_dir> [--fps 4] [--out-dir videos/]
    python tiff_to_video.py <tiff_dir> --combined-only       # only the 4-panel video
    python tiff_to_video.py <tiff_dir> --no-combined         # skip the 4-panel video

Example:
    python tiff_to_video.py forecast_apr18/ --fps 4
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine as RasterioAffine
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd

# Radar palette (same as main_optimized.py)
from palette_radar_35 import RAIN_CLASSES as RADAR_35_CLASSES, MAX_THRESHOLD as RADAR_35_MAX

# ---------------------------------------------------------------------------
# Natural Earth data caches
# ---------------------------------------------------------------------------
_borders_gdf = None
_provinces_gdf = None
_cities_gdf = None

NE_BASE = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
)
NE_COUNTRIES_URL = f"{NE_BASE}/ne_50m_admin_0_countries.geojson"
NE_PROVINCES_URL = f"{NE_BASE}/ne_10m_admin_1_states_provinces.geojson"
NE_CITIES_URL = f"{NE_BASE}/ne_10m_populated_places.geojson"


def get_europe_borders() -> "geopandas.GeoDataFrame":
    """Load and cache Natural Earth country borders for Europe."""
    global _borders_gdf
    if _borders_gdf is not None:
        return _borders_gdf

    print("Downloading Natural Earth country borders …")
    gdf = gpd.read_file(NE_COUNTRIES_URL)
    europe = gdf.cx[-12:40, 30:72]
    _borders_gdf = europe
    return europe


def get_europe_provinces() -> "geopandas.GeoDataFrame":
    """Load and cache state/province/department boundaries for Europe."""
    global _provinces_gdf
    if _provinces_gdf is not None:
        return _provinces_gdf

    print("Downloading Natural Earth provinces / departments …")
    gdf = gpd.read_file(NE_PROVINCES_URL)
    europe = gdf.cx[-12:40, 30:72]
    _provinces_gdf = europe
    return europe


def get_europe_cities() -> "geopandas.GeoDataFrame":
    """Load and cache populated places for Europe."""
    global _cities_gdf
    if _cities_gdf is not None:
        return _cities_gdf

    print("Downloading Natural Earth populated places …")
    gdf = gpd.read_file(NE_CITIES_URL)
    europe = gdf.cx[-12:40, 30:72]
    _cities_gdf = europe
    return europe


# ---------------------------------------------------------------------------
# Crop presets (lon_min, lat_min, lon_max, lat_max)
# ---------------------------------------------------------------------------

CROP_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "paris":   (1.0, 48.0, 4.0, 49.5),
    "idf":     (1.0, 48.0, 4.0, 49.5),       # Île-de-France (same as paris)
    "france":  (-5.5, 41.0, 10.0, 51.5),
    "benelux": (2.0, 48.5, 8.0, 54.0),
    "uk":      (-7.0, 49.5, 2.5, 59.0),
    "se_france": (2.0, 42.5, 8.0, 46.5),   # South-East France (PACA, Auvergne-Rhône-Alpes, Occitanie est)
}


def parse_crop_option(crop_str: str) -> tuple[float, float, float, float] | None:
    """
    Parse --crop argument. Accepts:
      - A preset name: 'paris', 'france', 'idf', …
      - Custom bounds: 'lon_min,lat_min,lon_max,lat_max' (e.g. '1.5,48.2,3.5,49.2')
    Returns (lon_min, lat_min, lon_max, lat_max) or None.
    """
    crop_str = crop_str.strip().lower()
    if crop_str in CROP_PRESETS:
        return CROP_PRESETS[crop_str]
    parts = crop_str.split(",")
    if len(parts) == 4:
        try:
            return tuple(float(p) for p in parts)
        except ValueError:
            pass
    print(f"Error: invalid --crop value '{crop_str}'. Use a preset ({', '.join(CROP_PRESETS)}) "
          f"or lon_min,lat_min,lon_max,lat_max", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Geo-transform helpers
# ---------------------------------------------------------------------------

def lonlat_to_pixel(lon, lat, transform) -> tuple[int, int]:
    """Convert lon/lat to pixel (col, row) using an Affine transform."""
    # transform * (col, row) = (x, y)  →  we need the inverse
    inv = ~transform
    x, y = inv * (lon, lat)
    return int(round(x)), int(round(y))


def crop_array(
    arr: np.ndarray,
    transform,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, "rasterio.Affine"]:
    """
    Crop array to the given lon/lat bounding box.
    Returns (cropped_array, new_transform).
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    inv = ~transform

    # Top-left = (lon_min, lat_max) → (col_start, row_start)
    col_start, row_start = inv * (lon_min, lat_max)
    # Bottom-right = (lon_max, lat_min) → (col_end, row_end)
    col_end, row_end = inv * (lon_max, lat_min)

    col_start = max(0, int(round(col_start)))
    row_start = max(0, int(round(row_start)))
    col_end = min(arr.shape[1], int(round(col_end)))
    row_end = min(arr.shape[0], int(round(row_end)))

    if col_end <= col_start or row_end <= row_start:
        print(f"Warning: crop region is outside the TIFF bounds, using full array", file=sys.stderr)
        return arr, transform

    cropped = arr[row_start:row_end, col_start:col_end]
    from rasterio.transform import Affine
    new_transform = Affine(transform.a, transform.b,
                           transform.c + col_start * transform.a,
                           transform.d, transform.e,
                           transform.f + row_start * transform.e)
    return cropped, new_transform


def draw_borders_on_frame(
    frame: np.ndarray,
    transform,
    gdf: "geopandas.GeoDataFrame",
    color: tuple[int, int, int] = (180, 180, 180),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw country borders on an RGB frame (H, W, 3) using the rasterio Affine
    transform to map lon/lat coordinates to pixels.
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    for geom in gdf.geometry:
        if geom.geom_type == "Polygon":
            rings = [geom.exterior]
        elif geom.geom_type == "MultiPolygon":
            rings = []
            for poly in geom.geoms:
                rings.append(poly.exterior)
        else:
            continue

        for ring in rings:
            coords = np.array(ring.coords)
            pts = []
            for lon, lat in coords:
                col, row = lonlat_to_pixel(lon, lat, transform)
                pts.append((col, row))
            # Filter to points roughly in frame (with margin)
            pts = [(x, y) for x, y in pts if -500 <= x < w + 500 and -500 <= y < h + 500]
            if len(pts) >= 2:
                cv2.polylines(result, [np.array(pts, dtype=np.int32)], False, color, thickness)

    return result


def draw_provinces_on_frame(
    frame: np.ndarray,
    transform,
    gdf: "geopandas.GeoDataFrame",
    color: tuple[int, int, int] = (120, 120, 140),
    thickness: int = 1,
) -> np.ndarray:
    """Draw province / department boundaries (thinner, dimmer than country borders)."""
    h, w = frame.shape[:2]
    result = frame.copy()

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            rings = [geom.exterior]
        elif geom.geom_type == "MultiPolygon":
            rings = []
            for poly in geom.geoms:
                rings.append(poly.exterior)
        elif geom.geom_type == "LineString":
            rings = [geom]
        elif geom.geom_type == "MultiLineString":
            rings = list(geom.geoms)
        else:
            continue

        for ring in rings:
            coords = np.array(ring.coords)
            pts = []
            for lon, lat in coords:
                col, row = lonlat_to_pixel(lon, lat, transform)
                pts.append((col, row))
            pts = [(x, y) for x, y in pts if -500 <= x < w + 500 and -500 <= y < h + 500]
            if len(pts) >= 2:
                cv2.polylines(result, [np.array(pts, dtype=np.int32)], False, color, thickness)

    return result


def draw_cities_on_frame(
    frame: np.ndarray,
    transform,
    gdf: "geopandas.GeoDataFrame",
    region_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """
    Draw cities as dots with name labels.  Population threshold adapts to zoom:
      - full Europe (>10° span): pop ≥ 500 000
      - country-level (~5°):      pop ≥ 100 000
      - region/city (~2°):        pop ≥ 30 000
      - zoomed-in (<1°):          pop ≥ 5 000
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # Estimate zoom level from frame width in degrees
    if region_bounds:
        lon_span = region_bounds[2] - region_bounds[0]
    else:
        lon_span = w * abs(transform.a)  # degrees per pixel * pixels

    if lon_span > 15:
        pop_threshold = 500_000
    elif lon_span > 5:
        pop_threshold = 100_000
    elif lon_span > 1.5:
        pop_threshold = 30_000
    else:
        pop_threshold = 5_000

    # Scale dot and text to frame
    s = max(w / 3583.0, 0.15)
    dot_radius = max(2, round(4 * s))
    font_scale = max(0.25, 0.5 * s)
    thickness = max(1, round(1 * s))

    # Filter cities
    if region_bounds:
        lon_min, lat_min, lon_max, lat_max = region_bounds
        cities = gdf.cx[lon_min:lon_max, lat_min:lat_max]
    else:
        cities = gdf

    for _, row in cities.iterrows():
        pop = row.get("POP_MAX", row.get("pop_max", 0))
        if isinstance(pop, (int, float)) and pop < pop_threshold:
            continue

        geom = row.geometry
        if geom is None:
            continue
        lon, lat = geom.x, geom.y
        cx, cy = lonlat_to_pixel(lon, lat, transform)
        if not (0 <= cx < w and 0 <= cy < h):
            continue

        # Dot
        cv2.circle(result, (cx, cy), dot_radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result, (cx, cy), dot_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Label
        name = row.get("NAME", row.get("name", ""))
        if name:
            cv2.putText(
                result, name, (cx + dot_radius + 3, cy + round(4 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )

    return result


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

def apply_lightning_colormap(data_uint8: np.ndarray, vmax: float) -> np.ndarray:
    """
    Lightning: yellow (low) -> orange -> red (high), matching LIGHTNING_CMAP
    in main_optimized.py. Transparent black for zero / nodata.
    Returns RGBA uint8 (H, W, 4).
    """
    h, w = data_uint8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    non_zero = data_uint8 > 0

    # Scale: 0-255 mapped to 0-vmax
    # Build a continuous yellow→red colormap
    cmap = LinearSegmentedColormap.from_list(
        "lightning",
        [(0, (1, 1, 0)), (0.25, (1, 1, 0)), (0.5, (1, 0.8, 0)), (0.75, (1, 0.4, 0)), (1, (1, 0, 0))],
    )

    normalized = data_uint8.astype(np.float32) / 255.0
    colored = cmap(normalized)  # (H, W, 4) float
    colored = (colored * 255).astype(np.uint8)

    rgba[non_zero] = colored[non_zero]
    return rgba


# ---------------------------------------------------------------------------
# Radar LUT (matching main_optimized.py exactly)
# ---------------------------------------------------------------------------
_RADAR_MAX_RATE = float(RADAR_35_MAX)
_RADAR_LOG_MIN = np.log(0.005)   # floor of log range
_RADAR_LOG_MAX = np.log(_RADAR_MAX_RATE)
_radar_thresholds = np.array([rc.threshold for rc in RADAR_35_CLASSES])
_radar_rgbs = np.array([list(rc.rgb) + [255] for rc in RADAR_35_CLASSES], dtype=np.uint8)

_RADAR_CMAP_LUT = np.zeros((256, 4), dtype=np.uint8)
for _i in range(256):
    _rate = np.exp(_RADAR_LOG_MIN + (_i / 255.0) * (_RADAR_LOG_MAX - _RADAR_LOG_MIN))
    if _rate < RADAR_35_CLASSES[0].threshold:
        continue
    _idx = int(np.searchsorted(_radar_thresholds, _rate, side='right')) - 1
    _idx = max(0, min(_idx, len(_radar_thresholds) - 1))
    _RADAR_CMAP_LUT[_i] = _radar_rgbs[_idx]


def apply_radar_colormap(data_uint8: np.ndarray) -> np.ndarray:
    """
    Radar: same palette_radar_35 colorbar as main_optimized.py.
    Transparent for zero / nodata.
    Returns RGBA uint8 (H, W, 4).
    """
    h, w = data_uint8.shape
    rgba = _RADAR_CMAP_LUT[data_uint8.ravel()].reshape(h, w, 4).copy()

    non_zero = data_uint8 > 0
    rgba[~non_zero] = [0, 0, 0, 0]
    return rgba


def apply_ir_colormap(data_uint8: np.ndarray) -> np.ndarray:
    """
    IR (sat_ch1): red colormap (dark → bright red → white).
    Transparent for zero / nodata.
    Returns RGBA uint8 (H, W, 4).
    """
    h, w = data_uint8.shape

    cmap = LinearSegmentedColormap.from_list(
        "ir_red",
        [
            (0.0, (0, 0, 0)),
            (0.15, (0.3, 0, 0)),
            (0.35, (0.6, 0, 0)),
            (0.55, (0.9, 0.1, 0)),
            (0.75, (1, 0.4, 0.1)),
            (0.9, (1, 0.7, 0.3)),
            (1.0, (1, 1, 1)),
        ],
    )

    normalized = data_uint8.astype(np.float32) / 255.0
    colored = cmap(normalized)
    colored = (colored * 255).astype(np.uint8)

    non_zero = data_uint8 > 0
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[non_zero] = colored[non_zero]
    return rgba


def apply_default_colormap(data_uint8: np.ndarray) -> np.ndarray:
    """Fallback: viridis colormap. Returns RGBA uint8."""
    normalized = data_uint8.astype(np.float32) / 255.0
    cmap = plt.get_cmap("viridis")
    colored = (cmap(normalized) * 255).astype(np.uint8)
    colored[:, :, 3] = 255
    non_zero = data_uint8 > 0
    colored[~non_zero] = [0, 0, 0, 255]
    return colored


# ---------------------------------------------------------------------------
# Data loading & normalization
# ---------------------------------------------------------------------------

def load_tiff(path: Path) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Load TIFF band 1 as float32, return (array, transform)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
    return arr, transform


def normalize_to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Clip to [vmin, vmax] then scale to 0-255 uint8."""
    clipped = np.clip(arr, vmin, vmax)
    if vmax > vmin:
        normalized = (clipped - vmin) / (vmax - vmin) * 255.0
    else:
        normalized = np.zeros_like(clipped)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized.astype(np.uint8)


def normalize_radar_to_uint8(arr_dbz: np.ndarray) -> np.ndarray:
    """
    Radar-specific normalization matching main_optimized.py:
    dBZ → Z (linear) → rain rate (mm/h) via Marshall-Palmer → log mapping → uint8.
    This produces indices into _RADAR_CMAP_LUT.
    """
    rain_rate = np.zeros_like(arr_dbz)
    valid = arr_dbz > 0
    z_linear = np.power(10.0, arr_dbz[valid] / 10.0)
    rain_rate[valid] = np.power(z_linear / 200.0, 1.0 / 1.6)

    log_rate = np.log(np.clip(rain_rate, 0.01, _RADAR_MAX_RATE))
    data_norm = np.clip(
        (log_rate - _RADAR_LOG_MIN) / (_RADAR_LOG_MAX - _RADAR_LOG_MIN),
        0, 1,
    )
    indices = (data_norm * 255).astype(np.uint8)
    return indices


# ---------------------------------------------------------------------------
# Channel config
# ---------------------------------------------------------------------------

CHANNEL_CONFIG = {
    "lightning": {"vmin": 0, "vmax": 4, "cmap": "lightning", "borders": True},
    "radar":     {"vmin": 0, "vmax": 75, "cmap": "radar",     "borders": True},
    "sat_ch0":   {"vmin": 0, "vmax": 12, "cmap": "default",   "borders": False},
    "sat_ch1":   {"vmin": 3, "vmax": 120, "cmap": "ir",       "borders": False},
    "sat_ch2":   {"vmin": -3, "vmax": 120, "cmap": "ir",      "borders": False},
}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def collect_channels(tiff_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """
    Returns a dict mapping channel_key -> sorted list of (timestamp, path).
    Channel keys: "sat_ch0", "sat_ch1", "radar", "lightning", etc.
    """
    pattern = re.compile(r"forecast_(\d{12})_(.+)\.tiff$")
    channels: dict[str, list] = defaultdict(list)

    for f in tiff_dir.glob("forecast_*.tiff"):
        m = pattern.match(f.name)
        if m:
            ts, ch_key = m.group(1), m.group(2)
            channels[ch_key].append((ts, f))

    for ch_key in channels:
        channels[ch_key].sort(key=lambda x: x[0])

    return dict(channels)


# ---------------------------------------------------------------------------
# Timestamp overlay
# ---------------------------------------------------------------------------

def format_timestamp(ts: str) -> str:
    """Convert '202604181800' → '2026-04-18 18:00 UTC'."""
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[8:10]}:{ts[10:12]} UTC"


CHANNEL_DISPLAY = {
    "lightning": "Lightning",
    "radar": "Radar Reflectivity",
    "sat_ch0": "Satellite VIS (Ch0)",
    "sat_ch1": "Satellite IR (Ch1)",
    "sat_ch2": "Satellite (Ch2)",
}


def draw_timestamp_overlay(
    frame: np.ndarray,
    ts: str,
    channel_label: str,
) -> np.ndarray:
    """
    Draw a semi-transparent info box in the top-left corner showing:
      - Channel name
      - Forecast time
    Text size scales with frame width (reference: 3583 px = full Europe).
    """
    h, w = frame.shape[:2]
    result = frame.copy()

    # Scale factor relative to the full Europe frame (3583 px wide)
    s = max(w / 3583.0, 0.15)
    font_scale = 1.2 * s
    thickness = max(1, round(2 * s))
    padding = max(4, round(12 * s))

    time_str = format_timestamp(ts)
    label = channel_label

    # Measure text sizes
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    (tw, th), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, thickness)

    box_w = max(lw, tw) + padding * 2
    box_h = lh + th + padding * 3 + 4  # 4px gap between lines

    # Semi-transparent dark background box
    overlay = result.copy()
    cv2.rectangle(overlay, (padding, padding), (padding + box_w, padding + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, result, 0.35, 0, result)

    # White text
    y_label = padding + lh + padding // 2
    y_time = y_label + th + 6
    cv2.putText(result, label, (padding + padding // 2, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(result, time_str, (padding + padding // 2, y_time),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (220, 220, 220), thickness, cv2.LINE_AA)

    return result


def draw_panel_label(
    frame: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    Draw a small semi-transparent label in the top-left corner of a panel.
    Used for the combined 4-panel video.
    Text size scales with frame width.
    """
    h, w = frame.shape[:2]
    s = max(w / 3583.0, 0.15)
    font_scale = 1.8 * s
    thickness = max(1, round(2 * s))
    padding = max(5, round(10 * s))

    result = frame.copy()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    box_w = tw + padding * 2
    box_h = th + padding * 2

    overlay = result.copy()
    cv2.rectangle(overlay, (padding, padding), (padding + box_w, padding + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    cv2.putText(result, label, (padding + padding // 2, padding + th + padding // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return result


# ---------------------------------------------------------------------------
# Video writing (RGBA → BGR via dark background compositing)
# ---------------------------------------------------------------------------

def rgba_to_bgr(rgba: np.ndarray, bg_color: tuple[int, int, int] = (10, 10, 30)) -> np.ndarray:
    """Composite RGBA onto a solid background and return BGR for OpenCV."""
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.array(bg_color, dtype=np.float32)
    rgb = rgba[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # RGB → BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


MIN_OUTPUT_WIDTH = 1920


def upscale_frame(frame: np.ndarray, target_width: int = MIN_OUTPUT_WIDTH) -> np.ndarray:
    """Upscale a BGR frame to at least target_width using Lanczos interpolation."""
    h, w = frame.shape[:2]
    if w >= target_width:
        return frame
    scale = target_width / w
    new_w = target_width
    new_h = max(1, round(h * scale))
    # Use INTER_LANCZOS4 for sharp upscaling
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
    for frame in frames:
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_channel_frame(
    arr: np.ndarray,
    transform,
    ch_key: str,
    cfg: dict,
    borders_gdf,
    provinces_gdf=None,
    cities_gdf=None,
    crop_bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Process a single channel array into a BGR frame (colormap + borders + provinces + cities)."""
    cmap_type = cfg["cmap"]

    if cmap_type == "radar":
        # Radar uses dBZ → rain rate → log → LUT (matching main_optimized.py)
        data_uint8 = normalize_radar_to_uint8(arr)
        rgba = apply_radar_colormap(data_uint8)
    else:
        data_uint8 = normalize_to_uint8(arr, cfg["vmin"], cfg["vmax"])
        if cmap_type == "lightning":
            rgba = apply_lightning_colormap(data_uint8, cfg["vmax"])
        elif cmap_type == "ir":
            rgba = apply_ir_colormap(data_uint8)
        else:
            rgba = apply_default_colormap(data_uint8)

    bgr = rgba_to_bgr(rgba)

    if cfg["borders"] and borders_gdf is not None:
        border_color = (180, 180, 180) if cmap_type == "lightning" else (150, 150, 150)
        bgr = draw_borders_on_frame(bgr, transform, borders_gdf, color=border_color, thickness=1)

    # Upscale small frames BEFORE drawing provinces/cities so they stay crisp
    bgr = upscale_frame(bgr)

    # Recompute transform for the upscaled frame
    s_up = bgr.shape[1] / arr.shape[1]
    up_transform = RasterioAffine(
        transform.a / s_up, transform.b,
        transform.c,
        transform.d, transform.e / s_up,
        transform.f,
    )

    if cfg["borders"] and provinces_gdf is not None:
        bgr = draw_provinces_on_frame(bgr, up_transform, provinces_gdf, color=(100, 100, 120), thickness=1)

    if cfg["borders"] and cities_gdf is not None:
        bgr = draw_cities_on_frame(bgr, up_transform, cities_gdf, region_bounds=crop_bounds)

    return bgr


def make_combined_frame(
    channel_frames: dict[str, np.ndarray],
    ts: str,
    layout_channels: list[str],
) -> np.ndarray:
    """
    Compose a 2x2 grid frame from per-channel BGR images.
    Adds a title bar with the timestamp and per-panel labels.
    Text size scales with frame width.
    """
    # All frames should be same size; pick first as reference
    ref = next(iter(channel_frames.values()))
    h, w = ref.shape[:2]

    # Scale layout to frame size (reference: 3583 px = full Europe)
    s = max(w / 3583.0, 0.15)
    title_height = max(16, round(80 * s))
    gap = max(2, round(4 * s))
    title_font_scale = 1.8 * s
    title_thickness = max(1, round(3 * s))

    cols = 2
    rows = 2
    canvas_w = cols * w + (cols - 1) * gap
    canvas_h = rows * h + (rows - 1) * gap + title_height

    canvas = np.full((canvas_h, canvas_w, 3), (10, 10, 30), dtype=np.uint8)

    # Title bar with timestamp
    time_str = format_timestamp(ts)
    title_text = f"FlashNet Forecast  —  {time_str}"
    (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)
    tx = (canvas_w - tw) // 2
    ty = title_height - (title_height - th) // 2 - max(2, round(4 * s))
    cv2.putText(canvas, title_text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)

    # Place panels
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (col, row)
    for idx, ch_key in enumerate(layout_channels):
        col, row = positions[idx]
        frame = channel_frames.get(ch_key)
        if frame is None:
            frame = np.full((h, w, 3), (20, 20, 40), dtype=np.uint8)

        x = col * (w + gap)
        y = row * (h + gap) + title_height
        canvas[y:y + h, x:x + w] = frame

        # Per-panel label
        label = CHANNEL_DISPLAY.get(ch_key, ch_key)
        panel = canvas[y:y + h, x:x + w]
        panel_labeled = draw_panel_label(panel, label)
        canvas[y:y + h, x:x + w] = panel_labeled

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Convert forecast TIFFs to videos with colormaps and borders")
    parser.add_argument("tiff_dir", type=Path, help="Directory containing forecast TIFF files")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second (default: 4)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as tiff_dir)")
    parser.add_argument("--combined-only", action="store_true", help="Only produce the combined 4-panel video")
    parser.add_argument("--no-combined", action="store_true", help="Skip the combined 4-panel video")
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help=(
            "Crop to a region. Use a preset name: "
            + ", ".join(f"'{k}'" for k in CROP_PRESETS)
            + " — or custom bounds: lon_min,lat_min,lon_max,lat_max"
        ),
    )
    args = parser.parse_args()

    tiff_dir = args.tiff_dir.resolve()
    if not tiff_dir.is_dir():
        print(f"Error: {tiff_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir.resolve() if args.out_dir else tiff_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = collect_channels(tiff_dir)
    if not channels:
        print("No forecast TIFF files found.", file=sys.stderr)
        sys.exit(1)

    # Resolve crop region
    crop_bounds = parse_crop_option(args.crop) if args.crop else None

    # Pre-load European borders once
    borders_gdf = get_europe_borders()
    provinces_gdf = get_europe_provinces()
    cities_gdf = get_europe_cities()

    if crop_bounds:
        print(f"Crop region: lon [{crop_bounds[0]}, {crop_bounds[2]}], "
              f"lat [{crop_bounds[1]}, {crop_bounds[3]}]")

    # ------------------------------------------------------------------
    # Determine common set of timestamps across channels
    # ------------------------------------------------------------------
    all_timestamps: set[str] = set()
    for entries in channels.values():
        ts_set = {ts for ts, _ in entries}
        all_timestamps |= ts_set
    sorted_timestamps = sorted(all_timestamps)

    # Build lookup: (ts, ch_key) -> path
    ts_channel_map: dict[tuple[str, str], Path] = {}
    for ch_key, entries in channels.items():
        for ts, path in entries:
            ts_channel_map[(ts, ch_key)] = path

    # Define layout order for the combined video
    combined_layout = ["lightning", "radar", "sat_ch0", "sat_ch1"]

    # Cache loaded arrays for reuse between per-channel and combined
    frame_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}  # (ts, ch_key) -> (arr, transform)

    # ------------------------------------------------------------------
    # 1) Per-channel videos
    # ------------------------------------------------------------------
    if not args.combined_only:
        for ch_key, entries in sorted(channels.items()):
            cfg = CHANNEL_CONFIG.get(ch_key, {"vmin": 0, "vmax": 100, "cmap": "default", "borders": False})
            print(f"[{ch_key}] {len(entries)} frames  (colormap={cfg['cmap']}, borders={cfg['borders']})")

            frames_bgr: list[np.ndarray] = []

            for idx, (ts, path) in enumerate(entries):
                arr, transform = load_tiff(path)
                if crop_bounds:
                    arr, transform = crop_array(arr, transform, crop_bounds)
                if idx == 0:
                    print(f"  frame size: {arr.shape[1]}x{arr.shape[0]} px")
                frame_cache[(ts, ch_key)] = (arr, transform)

                bgr = build_channel_frame(arr, transform, ch_key, cfg, borders_gdf,
                                             provinces_gdf, cities_gdf, crop_bounds)

                # Draw timestamp + channel label overlay
                ch_label = CHANNEL_DISPLAY.get(ch_key, ch_key)
                bgr = draw_timestamp_overlay(bgr, ts, ch_label)

                frames_bgr.append(bgr)

                if (idx + 1) % 20 == 0:
                    print(f"  processed {idx + 1}/{len(entries)} frames")

            out_path = out_dir / f"forecast_{ch_key}{'_' + args.crop if args.crop else ''}.mp4"
            write_video(frames_bgr, out_path, args.fps)
            print(f"  -> {out_path}  ({len(frames_bgr)} frames, {args.fps} fps)")

    # ------------------------------------------------------------------
    # 2) Combined 4-panel video
    # ------------------------------------------------------------------
    if not args.no_combined:
        print("\n[combined] Building 4-panel video …")

        combined_frames: list[np.ndarray] = []
        total = len(sorted_timestamps)

        for idx, ts in enumerate(sorted_timestamps):
            channel_frames: dict[str, np.ndarray] = {}
            for ch_key in combined_layout:
                path = ts_channel_map.get((ts, ch_key))
                if path is None:
                    continue
                cfg = CHANNEL_CONFIG.get(ch_key, {"vmin": 0, "vmax": 100, "cmap": "default", "borders": False})

                if (ts, ch_key) in frame_cache:
                    arr, transform = frame_cache[(ts, ch_key)]
                else:
                    arr, transform = load_tiff(path)
                    if crop_bounds:
                        arr, transform = crop_array(arr, transform, crop_bounds)
                    frame_cache[(ts, ch_key)] = (arr, transform)

                bgr = build_channel_frame(arr, transform, ch_key, cfg, borders_gdf,
                                             provinces_gdf, cities_gdf, crop_bounds)
                channel_frames[ch_key] = bgr

            if channel_frames:
                combined = make_combined_frame(channel_frames, ts, combined_layout)
                combined_frames.append(combined)

            if (idx + 1) % 20 == 0:
                print(f"  processed {idx + 1}/{total} timestamps")

        if combined_frames:
            out_path = out_dir / f"forecast_combined{'_' + args.crop if args.crop else ''}.mp4"
            write_video(combined_frames, out_path, args.fps)
            print(f"  -> {out_path}  ({len(combined_frames)} frames, {args.fps} fps)")
        else:
            print("  No frames for combined video.")


if __name__ == "__main__":
    main()
