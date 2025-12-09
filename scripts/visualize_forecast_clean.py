import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import sys
from datetime import datetime, timedelta
import imageio
import matplotlib.pyplot as plt
import h5py
import yaml
import cartopy.crs as ccrs
from cartopy.feature import BORDERS, COASTLINE

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

# Load config
config_path = os.path.join(project_root, "meteolibre_model/config/configs.yml")
with open(config_path) as f:
    config = yaml.safe_load(f)
params = config['model_v9_mtg_world_lightning_shortcut']
context_frames = params['model']['context_frames']

c_sat = params['model']['sat_in_channels']
c_lightning = params['model']['kpi_in_channels']
nb_channels = c_sat + c_lightning

# Define geospatial extents for Europe/Africa scale (adjust if HDF5 attrs provide exact bounds)
LON_MIN, LON_MAX = -10, 33
LAT_MIN, LAT_MAX = 33, 65

def create_video(forecast_dir, data_file, output_dir, forecast_steps, use_region=False,
                 region_start_row=None, region_end_row=None, region_start_col=None, region_end_col=None):
    """
    Generates videos comparing forecasts to ground truth for each channel.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load h5 data
    with h5py.File(data_file, 'r') as hf:

        num_frames = hf.attrs['num_frames']

        sat_data = hf['sat_data'][:]
        lightning_data = hf['lightning_data'][:]

        elevation_data = hf["elevation_data"][:]
        elev_frame = elevation_data[None, None, :, :].repeat(num_frames, axis=0)  # (nbframe, 1, H, W)
        
        sat_data = np.concatenate([sat_data, elev_frame], axis=1)


    # Apply region focus if specified
    if use_region and all(x is not None for x in [region_start_row, region_end_row, region_start_col, region_end_col]):
        print(f"Focusing on region: rows {region_start_row}:{region_end_row}, cols {region_start_col}:{region_end_col}")
        full_height, full_width = sat_data.shape[2], sat_data.shape[3]  # Get full grid size before cropping
        # Crop the data arrays
        print(sat_data.shape)

        sat_data = sat_data[:, :, region_start_row:region_end_row, region_start_col:region_end_col]
        lightning_data = lightning_data[:, :, region_start_row:region_end_row, region_start_col:region_end_col]

        # Calculate adjusted extents for cropped region
        crop_height_ratio = (region_end_row - region_start_row) / full_height
        crop_width_ratio = (region_end_col - region_start_col) / full_width
        lat_extent = (LAT_MAX - LAT_MIN) * crop_height_ratio
        lon_extent = (LON_MAX - LON_MIN) * crop_width_ratio
        LAT_MIN_crop = LAT_MIN + (LAT_MAX - LAT_MIN) * (region_start_row / full_height)
        LON_MIN_crop = LON_MIN + (LON_MAX - LON_MIN) * (region_start_col / full_width)
        LAT_MAX_crop = LAT_MIN_crop + lat_extent
        LON_MAX_crop = LON_MIN_crop + lon_extent
    else:
        LAT_MIN_crop, LAT_MAX_crop = LAT_MIN, LAT_MAX
        LON_MIN_crop, LON_MAX_crop = LON_MIN, LON_MAX

    # Parse initial_date from filename
    filename = os.path.basename(data_file)

    date_str = filename.split('_full.h5')[0]  # "2025-10-14_04-00"
    date_part, time_part = date_str.split('_')[0], date_str.split('_')[1]

    year, month, day = map(int, date_part.split('-'))
    hour, minute = map(int, time_part.split('-'))
    initial_date = datetime(year, month, day, hour, minute) - timedelta(minutes=18*10) # 18 to be modify

    for channel in [0, 11, 17]:
        images = []

        vmin=-4 #min(np.min(forecast_channel_data), np.min(true_channel_data))
        vmax=None

        for step in range(forecast_steps):

            prediction_date = initial_date + timedelta(minutes=10 * (step + 1))

            # Load forecast data
            forecast_file = os.path.join(forecast_dir, f"forecast_{prediction_date.strftime('%Y%m%d%H%M')}.npz")
            if not os.path.exists(forecast_file):
                print(f"Warning: Forecast file not found for {prediction_date}, skipping.")
                continue

            forecast_data = np.load(forecast_file)
            sat_forecast = forecast_data['sat_forecast']
            lightning_forecast = forecast_data['lightning_forecast']

            # Apply region focus to forecast data if specified (already handled globally, but ensure consistency)
            if use_region and all(x is not None for x in [region_start_row, region_end_row, region_start_col, region_end_col]):
                sat_forecast = sat_forecast[:, region_start_row:region_end_row, region_start_col:region_end_col]
                lightning_forecast = lightning_forecast[:, region_start_row:region_end_row, region_start_col:region_end_col]

            forecast_full = np.concatenate([sat_forecast, lightning_forecast], axis=0)  # (nb_channels, H, W)
            
            forecast_channel_data = forecast_full[channel]

            # Load true data
            true_index = int((prediction_date - initial_date).total_seconds() / 600)
            if true_index >= num_frames:
                print(f"Warning: True index {true_index} out of range, skipping.")
                continue

            sat_true = sat_data[true_index]
            lightning_true = lightning_data[true_index]
            true_full = np.concatenate([sat_true, lightning_true], axis=0)
            true_channel_data = true_full[channel]

            if not vmax:
                vmax = max(np.max(forecast_channel_data), np.max(true_channel_data)) + 1.0

            # Generate a side-by-side comparison image with Cartopy (no colorbars, tight spacing)
            fig = plt.figure(figsize=(15, 5.5), facecolor='black')
            ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
            ax1 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

            # Plot forecast
            im0 = ax0.imshow(forecast_channel_data, cmap='viridis', vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree(), extent=[LON_MIN_crop, LON_MAX_crop, LAT_MIN_crop, LAT_MAX_crop])
            ax0.set_title(f'Forecast - Channel {channel} - {prediction_date.strftime("%Y-%m-%d %H:%M")}', color='white')
            #ax0.add_feature(COASTLINE, linewidth=1, edgecolor='gray', facecolor='none')
            #ax0.set_extent([LON_MIN_crop, LON_MAX_crop, LAT_MIN_crop, LAT_MAX_crop], crs=ccrs.PlateCarree())
            ax0.set_facecolor('black')

            cb = fig.colorbar(im0, ax=ax0)
            cb.remove()

            # Plot true
            im1 = ax1.imshow(true_channel_data, cmap='viridis', vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree(), extent=[LON_MIN_crop, LON_MAX_crop, LAT_MIN_crop, LAT_MAX_crop])
            ax1.set_title(f'True - Channel {channel} - {prediction_date.strftime("%Y-%m-%d %H:%M")}', color='white')
            #ax1.add_feature(COASTLINE, linewidth=1, edgecolor='gray', facecolor='none')
            #ax1.set_extent([LON_MIN_crop, LON_MAX_crop, LAT_MIN_crop, LAT_MAX_crop], crs=ccrs.PlateCarree())
            ax1.set_facecolor('black')

            cb = fig.colorbar(im1, ax=ax1)
            cb.remove()

            plt.subplots_adjust(wspace=0.02)  # Minimal spacing between subplots
            plt.tight_layout()

            # Convert plot to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            images.append(image[..., 1:])
            plt.close(fig)

        if images:
            video_path = os.path.join(output_dir, f'channel_{channel}_comparison.mp4')
            imageio.mimsave(video_path, images, fps=1)
            print(f'Saved video for channel {channel} to {video_path}')

def main():
    parser = argparse.ArgumentParser(description="Generate clean videos to compare forecasts with ground truth from H5 (no colorbars, tight side-by-side).")
    parser.add_argument(
        "--forecast_dir",
        type=str,
        default="generated_forecasts_shortcut",
        help="Directory containing the forecast .npz files.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../dataset/data_inference_full/2025-10-14_04-00_full.h5",
        help="Path to the HDF5 file containing the ground truth.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="forecast_videos_clean",
        help="Directory to save the generated videos.",
    )
    parser.add_argument(
        "--forecast_steps",
        type=int,
        default=12,
        help="Number of forecast steps to visualize.",
    )
    parser.add_argument(
        "--use_region",
        action="store_true",
        help="Enable focus on a specific region of the image.",
    )
    parser.add_argument(
        "--region_start_row",
        type=int,
        default=None,
        help="Start row index for region focus (inclusive).",
    )
    parser.add_argument(
        "--region_end_row",
        type=int,
        default=None,
        help="End row index for region focus (exclusive).",
    )
    parser.add_argument(
        "--region_start_col",
        type=int,
        default=None,
        help="Start column index for region focus (inclusive).",
    )
    parser.add_argument(
        "--region_end_col",
        type=int,
        default=None,
        help="End column index for region focus (exclusive).",
    )

    args = parser.parse_args()

    create_video(args.forecast_dir, args.data_file, args.output_dir, args.forecast_steps,
                 args.use_region, args.region_start_row, args.region_end_row,
                 args.region_start_col, args.region_end_col)

if __name__ == "__main__":
    main()