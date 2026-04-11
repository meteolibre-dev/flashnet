#!/usr/bin/env python3
"""
Convert forecast TIFF files into per-channel videos.

Usage:
    python tiff_to_video.py <output_dir> [--fps 4] [--out-dir videos/]
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio


def normalize_frame(arr: np.ndarray) -> np.ndarray:
    """Normalize array to uint8 [0, 255] using percentile clipping, ignoring NaNs."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    p2, p98 = np.nanpercentile(arr, (2, 98))
    clipped = np.clip(arr, p2, p98)
    if p98 > p2:
        normalized = (clipped - p2) / (p98 - p2) * 255
    else:
        normalized = np.zeros_like(clipped)
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized.astype(np.uint8)


def load_tiff(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


def collect_channels(tiff_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """
    Returns a dict mapping channel_key -> sorted list of (timestamp, path).
    Channel keys look like: "sat_ch0", "sat_ch1", "radar", "lightning".
    """
    # Pattern: forecast_{YYYYMMDDHHII}_{channel}.tiff
    pattern = re.compile(r"forecast_(\d{12})_(.+)\.tiff$")
    channels: dict[str, list] = defaultdict(list)

    for f in tiff_dir.glob("forecast_*.tiff"):
        m = pattern.match(f.name)
        if m:
            ts, ch_key = m.group(1), m.group(2)
            channels[ch_key].append((ts, f))

    # Sort each channel by timestamp
    for ch_key in channels:
        channels[ch_key].sort(key=lambda x: x[0])

    return dict(channels)


def write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Convert forecast TIFFs to videos")
    parser.add_argument("tiff_dir", type=Path, help="Directory containing forecast TIFF files")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
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

    for ch_key, entries in sorted(channels.items()):
        print(f"[{ch_key}] {len(entries)} frames")
        frames = []
        for ts, path in entries:
            arr = load_tiff(path)
            frame = normalize_frame(arr)
            frames.append(frame)

        out_path = out_dir / f"forecast_{ch_key}.mp4"
        write_video(frames, out_path, args.fps)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
