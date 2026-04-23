#!/usr/bin/env python3
"""
Convert forecast TIFF files into per-channel videos with proper colormaps
and European country borders overlay.

Colormaps (matching main_optimized.py):
  - lightning: yellow -> orange -> red
  - radar:     blue colormap
  - sat_ch1 (IR): red colormap

Usage:
    python tiff_to_video.py <tiff_dir> [--fps 4] [--out-dir videos/]

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
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd

# ---------------------------------------------------------------------------
# Natural Earth borders cache
# ---------------------------------------------------------------------------
_borders_gdf = None

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_admin_0_countries.geojson"
)


def get_europe_borders() -> "geopandas.GeoDataFrame":
    """Load and cache Natural Earth country borders for Europe."""
    global _borders_gdf
    if _borders_gdf is not None:
        return _borders_gdf

    print("Downloading Natural Earth country borders …")
    gdf = gpd.read_file(NATURAL_EARTH_URL)
    # Clip to our region of interest with a bit of margin
    europe = gdf.cx[-12:40, 30:72]
    _borders_gdf = europe
    return europe


# ---------------------------------------------------------------------------
# Geo-transform helpers
# ---------------------------------------------------------------------------

def lonlat_to_pixel(lon, lat, transform) -> tuple[int, int]:
    """Convert lon/lat to pixel (col, row) using an Affine transform."""
    # transform * (col, row) = (x, y)  →  we need the inverse
    inv = ~transform
    x, y = inv * (lon, lat)
    return int(round(x)), int(round(y))


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


def apply_radar_colormap(data_uint8: np.ndarray) -> np.ndarray:
    """
    Radar: blue colormap (dark blue → light blue → cyan → white).
    Transparent for zero / nodata.
    Returns RGBA uint8 (H, W, 4).
    """
    h, w = data_uint8.shape

    cmap = LinearSegmentedColormap.from_list(
        "radar_blue",
        [
            (0.0, (0, 0, 0.3)),
            (0.2, (0, 0.2, 0.6)),
            (0.4, (0, 0.4, 0.8)),
            (0.6, (0.1, 0.6, 0.9)),
            (0.8, (0.3, 0.8, 1.0)),
            (1.0, (0.7, 1.0, 1.0)),
        ],
    )

    normalized = data_uint8.astype(np.float32) / 255.0
    colored = cmap(normalized)
    colored = (colored * 255).astype(np.uint8)

    non_zero = data_uint8 > 0
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[non_zero] = colored[non_zero]
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
    font_scale: float = 1.2,
    thickness: int = 2,
    padding: int = 12,
) -> np.ndarray:
    """
    Draw a semi-transparent info box in the top-left corner showing:
      - Channel name
      - Forecast time
    """
    h, w = frame.shape[:2]
    result = frame.copy()

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

def main():
    parser = argparse.ArgumentParser(description="Convert forecast TIFFs to videos with colormaps and borders")
    parser.add_argument("tiff_dir", type=Path, help="Directory containing forecast TIFF files")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second (default: 4)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as tiff_dir)")
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

    # Pre-load European borders once
    borders_gdf = get_europe_borders()

    for ch_key, entries in sorted(channels.items()):
        cfg = CHANNEL_CONFIG.get(ch_key, {"vmin": 0, "vmax": 100, "cmap": "default", "borders": False})
        print(f"[{ch_key}] {len(entries)} frames  (colormap={cfg['cmap']}, borders={cfg['borders']})")

        frames_bgr: list[np.ndarray] = []
        reference_transform = None

        for idx, (ts, path) in enumerate(entries):
            arr, transform = load_tiff(path)
            if reference_transform is None:
                reference_transform = transform

            data_uint8 = normalize_to_uint8(arr, cfg["vmin"], cfg["vmax"])

            # Apply colormap
            cmap_type = cfg["cmap"]
            if cmap_type == "lightning":
                rgba = apply_lightning_colormap(data_uint8, cfg["vmax"])
            elif cmap_type == "radar":
                rgba = apply_radar_colormap(data_uint8)
            elif cmap_type == "ir":
                rgba = apply_ir_colormap(data_uint8)
            else:
                rgba = apply_default_colormap(data_uint8)

            # Convert to BGR for OpenCV
            bgr = rgba_to_bgr(rgba)

            # Draw borders for lightning and radar
            if cfg["borders"] and borders_gdf is not None:
                border_color = (180, 180, 180) if cmap_type == "lightning" else (150, 150, 150)
                bgr = draw_borders_on_frame(bgr, transform, borders_gdf, color=border_color, thickness=1)

            # Draw timestamp + channel label overlay
            ch_label = CHANNEL_DISPLAY.get(ch_key, ch_key)
            bgr = draw_timestamp_overlay(bgr, ts, ch_label)

            frames_bgr.append(bgr)

            if (idx + 1) % 20 == 0:
                print(f"  processed {idx + 1}/{len(entries)} frames")

        out_path = out_dir / f"forecast_{ch_key}.mp4"
        write_video(frames_bgr, out_path, args.fps)
        print(f"  -> {out_path}  ({len(frames_bgr)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
