#!/usr/bin/env python3
"""
Create denoising-trajectory videos from debug endpoint prediction TIFFs.

Each debug TIFF is the model's x-prediction (x0-hat) snapshot at a given
denoising step t.  This script orders them t=0.9 -> t=0.0 and renders one
video per channel plus a combined 2x2 panel, reusing the colormaps and
border overlays from tiff_to_video.py.

Usage:
    python debug_endpoint_video.py debug_data/debug_endpoints/
    python debug_endpoint_video.py debug_data/debug_endpoints/ --fps 2
    python debug_endpoint_video.py debug_data/debug_endpoints/ --combined-only
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Reuse everything from tiff_to_video.py
from tiff_to_video import (
    load_tiff,
    build_channel_frame,
    write_video,
    get_europe_borders,
    get_europe_provinces,
    get_europe_cities,
    CHANNEL_CONFIG,
    CHANNEL_DISPLAY,
    draw_panel_label,
    crop_array,
    parse_crop_option,
)


def collect_debug_channels(tiff_dir: Path) -> dict[str, list[tuple[float, str, Path]]]:
    """
    Returns dict: channel_key -> list of (t_value, timestamp, path).
    Parses filenames like:
        endpoint_t0.9_202605011700_sat_ch0.tiff
    """
    pattern = re.compile(r"endpoint_t(\d+\.\d+)_(\d{12})_(.+)\.tiff$")
    channels: dict[str, list] = defaultdict(list)

    for f in tiff_dir.glob("endpoint_t*_*.tiff"):
        m = pattern.match(f.name)
        if m:
            t_val = float(m.group(1))
            ts = m.group(2)
            ch_key = m.group(3)
            channels[ch_key].append((t_val, ts, f))

    # Sort by t DESCENDING (0.9 -> 0.0) so the video shows denoising progress
    for ch_key in channels:
        channels[ch_key].sort(key=lambda x: -x[0])

    return dict(channels)


def make_combined_frame_debug(
    channel_frames: dict[str, np.ndarray],
    t_val: float,
    layout_channels: list[str],
) -> np.ndarray:
    """Compose a 2x2 grid frame with a 't=X.X  endpoint prediction' title."""
    ref = next(iter(channel_frames.values()))
    h, w = ref.shape[:2]

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

    title_text = f"Endpoint prediction  —  t = {t_val:.1f}"
    (tw, th), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness)
    tx = (canvas_w - tw) // 2
    ty = title_height - (title_height - th) // 2 - max(2, round(4 * s))
    cv2.putText(canvas, title_text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for idx, ch_key in enumerate(layout_channels):
        col, row = positions[idx]
        frame = channel_frames.get(ch_key)
        if frame is None:
            frame = np.full((h, w, 3), (20, 20, 40), dtype=np.uint8)

        x = col * (w + gap)
        y = row * (h + gap) + title_height
        canvas[y:y + h, x:x + w] = frame

        label = CHANNEL_DISPLAY.get(ch_key, ch_key)
        panel = canvas[y:y + h, x:x + w]
        panel_labeled = draw_panel_label(panel, label)
        canvas[y:y + h, x:x + w] = panel_labeled

    return canvas


def draw_t_overlay(frame: np.ndarray, t_val: float, channel_label: str) -> np.ndarray:
    """Draw a semi-transparent info box: channel name + 't=0.X (endpoint prediction)'."""
    h, w = frame.shape[:2]
    result = frame.copy()

    s = max(w / 3583.0, 0.15)
    font_scale = 1.2 * s
    thickness = max(1, round(2 * s))
    padding = max(4, round(12 * s))

    label = channel_label
    time_str = f"t = {t_val:.1f}   (endpoint prediction x\u2080)"

    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    (tw, th), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, thickness)

    box_w = max(lw, tw) + padding * 2
    box_h = lh + th + padding * 3 + 4

    overlay = result.copy()
    cv2.rectangle(overlay, (padding, padding), (padding + box_w, padding + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, result, 0.35, 0, result)

    y_label = padding + lh + padding // 2
    y_time = y_label + th + 6
    cv2.putText(result, label, (padding + padding // 2, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(result, time_str, (padding + padding // 2, y_time),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (220, 220, 220), thickness, cv2.LINE_AA)

    return result


def main():
    parser = argparse.ArgumentParser(description="Create denoising-trajectory videos from debug endpoint TIFFs")
    parser.add_argument("tiff_dir", type=Path, help="Directory containing endpoint_t*_*.tiff files")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as tiff_dir)")
    parser.add_argument("--combined-only", action="store_true", help="Only produce the combined 4-panel video")
    parser.add_argument("--no-combined", action="store_true", help="Skip the combined 4-panel video")
    parser.add_argument("--crop", type=str, default=None,
                        help="Crop to a region preset (france, paris, ...) or lon_min,lat_min,lon_max,lat_max")
    args = parser.parse_args()

    tiff_dir = args.tiff_dir.resolve()
    if not tiff_dir.is_dir():
        print(f"Error: {tiff_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir.resolve() if args.out_dir else tiff_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = collect_debug_channels(tiff_dir)
    if not channels:
        print("No endpoint_t*_*.tiff files found.", file=sys.stderr)
        sys.exit(1)

    borders_gdf = get_europe_borders()
    provinces_gdf = get_europe_provinces()
    cities_gdf = get_europe_cities()

    crop_bounds = parse_crop_option(args.crop) if args.crop else None
    crop_suffix = f"_{args.crop}" if args.crop else ""
    if crop_bounds:
        print(f"Crop region: lon [{crop_bounds[0]}, {crop_bounds[2]}], lat [{crop_bounds[1]}, {crop_bounds[3]}]")

    # All t-values across channels
    all_t_values: set[float] = set()
    for entries in channels.values():
        all_t_values |= {t for t, _, _ in entries}
    sorted_t = sorted(all_t_values, reverse=True)  # 0.9 -> 0.0

    print(f"Found {len(sorted_t)} t-values: {sorted_t}")
    for ch_key, entries in sorted(channels.items()):
        print(f"  [{ch_key}] {len(entries)} frames")

    # Build lookup: (t_val, ch_key) -> path
    t_channel_map: dict[tuple[float, str], Path] = {}
    for ch_key, entries in channels.items():
        for t_val, ts, path in entries:
            t_channel_map[(t_val, ch_key)] = path

    combined_layout = ["lightning", "radar", "sat_ch0", "sat_ch1"]

    # Cache loaded arrays for reuse between per-channel and combined
    frame_cache: dict[tuple[float, str], tuple[np.ndarray, object]] = {}

    # ------------------------------------------------------------------
    # 1) Per-channel videos
    # ------------------------------------------------------------------
    if not args.combined_only:
        for ch_key, entries in sorted(channels.items()):
            cfg = CHANNEL_CONFIG.get(ch_key, {"vmin": 0, "vmax": 100, "cmap": "default", "borders": False})
            print(f"\n[{ch_key}] building denoising trajectory ({len(entries)} frames, t=0.9 -> 0.0)")

            frames_bgr: list[np.ndarray] = []

            for idx, (t_val, ts, path) in enumerate(entries):
                arr, transform = load_tiff(path)
                if crop_bounds:
                    arr, transform = crop_array(arr, transform, crop_bounds)
                if idx == 0:
                    print(f"  frame size: {arr.shape[1]}x{arr.shape[0]} px")
                frame_cache[(t_val, ch_key)] = (arr, transform)

                bgr = build_channel_frame(
                    arr, transform, ch_key, cfg,
                    borders_gdf, provinces_gdf, cities_gdf,
                    crop_bounds=crop_bounds,
                )

                ch_label = CHANNEL_DISPLAY.get(ch_key, ch_key)
                bgr = draw_t_overlay(bgr, t_val, ch_label)
                frames_bgr.append(bgr)

            out_path = out_dir / f"denoise_{ch_key}{crop_suffix}.mp4"
            write_video(frames_bgr, out_path, args.fps)
            print(f"  -> {out_path}  ({len(frames_bgr)} frames, {args.fps} fps)")

    # ------------------------------------------------------------------
    # 2) Combined 4-panel video
    # ------------------------------------------------------------------
    if not args.no_combined:
        print("\n[combined] Building 4-panel denoising trajectory ...")

        combined_frames: list[np.ndarray] = []

        for idx, t_val in enumerate(sorted_t):
            channel_frames: dict[str, np.ndarray] = {}
            for ch_key in combined_layout:
                path = t_channel_map.get((t_val, ch_key))
                if path is None:
                    continue
                cfg = CHANNEL_CONFIG.get(ch_key, {"vmin": 0, "vmax": 100, "cmap": "default", "borders": False})

                if (t_val, ch_key) in frame_cache:
                    arr, transform = frame_cache[(t_val, ch_key)]
                else:
                    arr, transform = load_tiff(path)
                    if crop_bounds:
                        arr, transform = crop_array(arr, transform, crop_bounds)
                    frame_cache[(t_val, ch_key)] = (arr, transform)

                bgr = build_channel_frame(
                    arr, transform, ch_key, cfg,
                    borders_gdf, provinces_gdf, cities_gdf,
                    crop_bounds=crop_bounds,
                )
                channel_frames[ch_key] = bgr

            if channel_frames:
                combined = make_combined_frame_debug(channel_frames, t_val, combined_layout)
                combined_frames.append(combined)

        if combined_frames:
            out_path = out_dir / f"denoise_combined{crop_suffix}.mp4"
            write_video(combined_frames, out_path, args.fps)
            print(f"  -> {out_path}  ({len(combined_frames)} frames, {args.fps} fps)")
        else:
            print("  No frames for combined video.")


if __name__ == "__main__":
    main()
