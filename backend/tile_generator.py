"""
Tile Generator for Pre-computed PNG Tiles

Generates PNG tiles from COG files for fast static serving.
Run this after TIFF files are created in the bucket.

Usage:
    python tile_generator.py --timestamp 202602231200 --bands lightning sat_ch0 sat_ch1

Or import as module:
    from tile_generator import generate_tiles_for_timestamp
    generate_tiles_for_timestamp("202602231200", ["lightning", "sat_ch0", "sat_ch1"])
"""

import os
import re
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
from PIL import Image
from io import BytesIO

try:
    from rio_tiler.io import Reader
    RIO_TILER_AVAILABLE = True
except ImportError:
    RIO_TILER_AVAILABLE = False
    print("Warning: rio-tiler not available")

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BUCKET_BASE_URL = os.getenv("BUCKET_BASE_URL", "gs://inference_result/forecasts")
TILES_BUCKET = os.getenv("TILES_BUCKET", "inference_result")
TILES_PREFIX = os.getenv("TILES_PREFIX", "tiles")

# European region bounds (for filtering tiles)
REGION_BOUNDS = {
    "lon_min": -10.0,
    "lat_min": 33.0,
    "lon_max": 33.0,
    "lat_max": 60.0,
}

# Zoom levels to generate
ZOOM_LEVELS = list(range(3, 9))  # 3 to 8 inclusive

# Band configuration (matching main_optimized.py)
BANDS = {
    "lightning": {"min": 0, "max": 4, "colormap": "custom", "invert": False},
    "sat_ch0": {"min": 0, "max": 12, "colormap": "viridis", "invert": False},
    "sat_ch1": {"min": 3, "max": 120, "colormap": "plasma", "invert": True},
    "sat_ch2": {"min": -3, "max": 120, "colormap": "plasma", "invert": True},
}

# Custom colormap for lightning
LIGHTNING_CMAP = {
    0: (255, 255, 0, 150),
    1: (255, 255, 0, 180),
    2: (255, 200, 0, 210),
    3: (255, 100, 0, 230),
    4: (255, 0, 0, 255),
}


def get_cog_url(timestamp: str, band: str) -> str:
    """Generate the COG URL for a given timestamp and band."""
    date_folder = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
    
    if BUCKET_BASE_URL.startswith("gs://"):
        return f"gs://inference_result/forecasts/{date_folder}/forecast_{timestamp}_{band}.tiff"
    else:
        return f"{BUCKET_BASE_URL}/{date_folder}/forecast_{timestamp}_{band}.tiff"


def lon_lat_to_tile(lon: float, lat: float, zoom: int) -> tuple:
    """Convert longitude/latitude to tile x/y at given zoom."""
    import math
    
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    return x, y


def get_tiles_for_region(bounds: dict, zoom: int) -> List[tuple]:
    """Get list of (x, y) tiles that cover the specified region."""
    x_min, y_min = lon_lat_to_tile(bounds["lon_min"], bounds["lat_max"], zoom)
    x_max, y_max = lon_lat_to_tile(bounds["lon_max"], bounds["lat_min"], zoom)
    
    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y))
    
    return tiles


def generate_tile_rgba(cog_url: str, x: int, y: int, z: int, band: str) -> Optional[np.ndarray]:
    """Generate RGBA array for a single tile from COG."""
    if not RIO_TILER_AVAILABLE:
        logger.error("rio-tiler not available")
        return None
    
    config = BANDS.get(band, {})
    if not config:
        return None
    
    try:
        with Reader(cog_url) as cog:
            tile = cog.tile(x, y, z, tilesize=256, indexes=(1,))
            data = tile.data[0].astype(np.float32)
            
            nodata_value = cog.nodata if hasattr(cog, 'nodata') else None
            
            nodata_mask = None
            if nodata_value is not None:
                nodata_mask = (data == nodata_value) | (~np.isfinite(data))
            elif tile.mask is not None:
                nodata_mask = ~tile.mask.astype(bool)
            elif np.any(~np.isfinite(data)):
                nodata_mask = ~np.isfinite(data)
            
            data = np.clip(data, config["min"], config["max"])
            data = ((data - config["min"]) / (config["max"] - config["min"]) * 255).astype(np.uint8)
            
            if band == "lightning":
                rgba = np.zeros((256, 256, 4), dtype=np.uint8)
                
                for val, color in LIGHTNING_CMAP.items():
                    if val > 0:
                        mask = data >= int(val * 255 / config["max"])
                        rgba[mask] = color
                
                non_zero_mask = data > 0
                rgba[non_zero_mask & (rgba[:, :, 3] == 0)] = [255, 255, 0, 150]
                
                if nodata_mask is not None:
                    rgba[nodata_mask] = [0, 0, 0, 0]
            else:
                from matplotlib import cm
                
                cmap = cm.get_cmap(config["colormap"])
                if config["invert"]:
                    cmap = cmap.reversed()
                
                normalized = data.astype(float) / 255.0
                rgba = (cmap(normalized) * 255).astype(np.uint8)
                rgba[:, :, 3] = 255
                
                zero_mask = data == 0
                rgba[zero_mask] = [0, 0, 0, 0]
                
                if nodata_mask is not None:
                    rgba[nodata_mask] = [0, 0, 0, 0]
            
            return rgba
            
    except Exception as e:
        logger.warning(f"Error generating tile {z}/{x}/{y}: {e}")
        return None


def tile_to_png(rgba: np.ndarray, format: str = "PNG") -> bytes:
    """Convert RGBA array to PNG bytes."""
    img = Image.fromarray(rgba, mode='RGBA')
    buffer = BytesIO()
    
    if format == "WEBP":
        img.save(buffer, format='WEBP', quality=85, method=6)
    else:
        img.save(buffer, format='PNG', optimize=True)
    
    buffer.seek(0)
    return buffer.getvalue()


def upload_to_gcs(bucket_name: str, blob_path: str, data: bytes) -> bool:
    """Upload data to GCS bucket."""
    if not GCS_AVAILABLE:
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(data, content_type=f"image/{'png' if blob_path.endswith('.png') else 'webp'}")
        return True
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        return False


def generate_tiles_for_band_timestamp(
    timestamp: str,
    band: str,
    zoom_levels: List[int] = ZOOM_LEVELS,
    region_bounds: dict = REGION_BOUNDS,
    bucket_name: str = TILES_BUCKET,
    prefix: str = TILES_PREFIX,
    output_format: str = "PNG",
    dry_run: bool = False
) -> dict:
    """
    Generate all tiles for a specific band and timestamp.
    
    Args:
        timestamp: Timestamp in YYYYMMDDHHMM format
        band: Band name (lightning, sat_ch0, sat_ch1, sat_ch2)
        zoom_levels: List of zoom levels to generate
        region_bounds: Geographic bounds for tile coverage
        bucket_name: GCS bucket for tiles
        prefix: Prefix for tile storage
        output_format: PNG or WEBP
        dry_run: If True, don't upload to GCS
    
    Returns:
        Dictionary with generation statistics
    """
    stats = {
        "timestamp": timestamp,
        "band": band,
        "tiles_generated": 0,
        "tiles_failed": 0,
        "tiles_skipped": 0,
        "zoom_levels": zoom_levels,
    }
    
    # Get COG URL
    cog_url = get_cog_url(timestamp, band)
    logger.info(f"Processing {band} for {timestamp} from {cog_url}")
    
    # Generate tiles for each zoom level
    for z in zoom_levels:
        tiles = get_tiles_for_region(region_bounds, z)
        logger.info(f"Zoom {z}: {len(tiles)} tiles to generate")
        
        for x, y in tiles:
            try:
                rgba = generate_tile_rgba(cog_url, x, y, z, band)
                
                if rgba is None:
                    stats["tiles_skipped"] += 1
                    continue
                
                png_data = tile_to_png(rgba, format=output_format)
                
                # Upload to GCS
                ext = "png" if output_format == "PNG" else "webp"
                blob_path = f"{prefix}/{band}/{timestamp}/{z}/{x}/{y}.{ext}"
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would upload: {blob_path} ({len(png_data)} bytes)")
                else:
                    success = upload_to_gcs(bucket_name, blob_path, png_data)
                    if success:
                        stats["tiles_generated"] += 1
                    else:
                        stats["tiles_failed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing tile {z}/{x}/{y}: {e}")
                stats["tiles_failed"] += 1
    
    logger.info(f"Completed {band}/{timestamp}: {stats['tiles_generated']} generated, "
                f"{stats['tiles_skipped']} skipped, {stats['tiles_failed']} failed")
    
    return stats


def generate_tiles_for_timestamp(
    timestamp: str,
    bands: List[str] = None,
    zoom_levels: List[int] = ZOOM_LEVELS,
    region_bounds: dict = REGION_BOUNDS,
    bucket_name: str = TILES_BUCKET,
    prefix: str = TILES_PREFIX,
    output_format: str = "PNG",
    max_workers: int = 4,
    dry_run: bool = False
) -> dict:
    """
    Generate all tiles for a timestamp across all bands.
    
    Args:
        timestamp: Timestamp in YYYYMMDDHHMM format
        bands: List of bands to process (default: all BANDS keys)
        zoom_levels: List of zoom levels to generate
        region_bounds: Geographic bounds for tile coverage
        bucket_name: GCS bucket for tiles
        prefix: Prefix for tile storage
        output_format: PNG or WEBP
        max_workers: Number of parallel workers for band processing
        dry_run: If True, don't upload to GCS
    
    Returns:
        Dictionary with generation statistics for all bands
    """
    if bands is None:
        bands = list(BANDS.keys())
    
    results = {
        "timestamp": timestamp,
        "total_bands": len(bands),
        "band_stats": [],
        "total_tiles": 0,
    }
    
    # Process bands in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for band in bands:
            if band not in BANDS:
                logger.warning(f"Unknown band: {band}, skipping")
                continue
            
            future = executor.submit(
                generate_tiles_for_band_timestamp,
                timestamp,
                band,
                zoom_levels,
                region_bounds,
                bucket_name,
                prefix,
                output_format,
                dry_run
            )
            futures[future] = band
        
        for future in as_completed(futures):
            band = futures[future]
            try:
                stats = future.result()
                results["band_stats"].append(stats)
                results["total_tiles"] += stats["tiles_generated"]
            except Exception as e:
                logger.error(f"Error processing band {band}: {e}")
    
    logger.info(f"Completed timestamp {timestamp}: {results['total_tiles']} total tiles generated")
    
    return results


def get_tile_url(band: str, timestamp: str, z: int, x: int, y: int, format: str = "png") -> str:
    """Get the URL for a pre-generated tile."""
    ext = format.lower()
    if format.lower() == "webp":
        ext = "webp"
    return f"gs://{TILES_BUCKET}/{TILES_PREFIX}/{band}/{timestamp}/{z}/{x}/{y}.{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-computed PNG tiles from COG files"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp in YYYYMMDDHHMM format (e.g., 202602231200)"
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=["lightning", "sat_ch0", "sat_ch1"],
        help="Bands to process"
    )
    parser.add_argument(
        "--zoom-levels",
        type=str,
        default="3-8",
        help="Zoom levels (e.g., 3-8 or 4,5,6)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="PNG",
        choices=["PNG", "WEBP"],
        help="Output format"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't upload to GCS, just log what would be done"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=TILES_BUCKET,
        help="GCS bucket for tiles"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=TILES_PREFIX,
        help="Prefix for tile storage"
    )
    
    args = parser.parse_args()
    
    # Parse zoom levels
    if "-" in args.zoom_levels:
        z_min, z_max = map(int, args.zoom_levels.split("-"))
        zoom_levels = list(range(z_min, z_max + 1))
    elif "," in args.zoom_levels:
        zoom_levels = [int(z) for z in args.zoom_levels.split(",")]
    else:
        zoom_levels = [int(args.zoom_levels)]
    
    logger.info(f"Starting tile generation for timestamp={args.timestamp}, "
                f"bands={args.bands}, zoom_levels={zoom_levels}")
    
    results = generate_tiles_for_timestamp(
        timestamp=args.timestamp,
        bands=args.bands,
        zoom_levels=zoom_levels,
        bucket_name=args.bucket,
        prefix=args.prefix,
        output_format=args.format,
        max_workers=args.workers,
        dry_run=args.dry_run
    )
    
    logger.info(f"Generation complete: {results}")
    
    # Print tile URLs for verification
    print("\nExample tile URLs:")
    for band in args.bands[:2]:
        for z in zoom_levels[:2]:
            tiles = get_tiles_for_region(REGION_BOUNDS, z)
            if tiles:
                x, y = tiles[0]
                url = get_tile_url(band, args.timestamp, z, x, y, args.format.lower())
                print(f"  {band} z={z}: {url}")


if __name__ == "__main__":
    main()
