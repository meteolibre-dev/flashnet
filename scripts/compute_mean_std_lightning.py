import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from meteolibre_model.dataset.dataset_mtg_lightning_radar import MeteoLibreMapDataset
from tqdm import tqdm

def compute_mean_std():
    """
    Computes the mean and standard deviation of the sat and lightning dataset channels, ignoring -10000 values.
    """
    # Load the local dataset
    localrepo = "/workspace/dataset"
    dataset = MeteoLibreMapDataset(localrepo=localrepo,
            cache_size=4,
        seed=44,
        nb_temporal=7)

    # Use a smaller subset for quick estimation, or iterate over the whole dataset
    # for a more accurate result. Set num_samples to None to iterate over everything.
    num_samples = 2000  # Or None

    print(f"Calculating mean and std over {num_samples or len(dataset)} samples, ignoring -10000...")

    # Initialize accumulators for sat data
    sat_total_sum = 0
    sat_total_sum_sq = 0
    sat_total_count = 0

    # Initialize accumulators for lightning data
    lightning_total_sum = 0
    lightning_total_sum_sq = 0
    lightning_total_count = 0

    for i, sample in enumerate(tqdm(dataset, total=num_samples)):
        if num_samples is not None and i >= num_samples:
            break

        # Process sat data
        sat_patch_data = sample["sat_patch_data"].numpy().astype(np.float32)

        # count the number of nan in the radar data
        isnan_radar = np.logical_or(np.isnan(sat_patch_data[:, -1, :, :]), sat_patch_data[:, -1, :, :] < -1.)

        sat_patch_data[:, -1, :, :] = np.where(isnan_radar, -10000.0, sat_patch_data[:, -1, :, :])

        sat_masked_patch_data = np.ma.masked_equal(sat_patch_data, -10000.0)

        if sat_total_sum is None:
            num_sat_channels = sat_patch_data.shape[1]
            sat_total_sum = np.zeros(num_sat_channels, dtype=np.float64)
            sat_total_sum_sq = np.zeros(num_sat_channels, dtype=np.float64)
            sat_total_count = np.zeros(num_sat_channels, dtype=np.int64)

        #breakpoint()

        sat_total_sum += sat_masked_patch_data.sum(axis=(0, 2, 3)).filled(0)
        sat_total_sum_sq += np.ma.power(sat_masked_patch_data, 2).sum(axis=(0, 2, 3)).filled(0)
        sat_total_count += sat_masked_patch_data.count(axis=(0, 2, 3))

        # Process lightning data
        lightning_patch_data = sample["lightning_patch_data"].numpy().astype(np.float32)
        lightning_masked_patch_data = np.ma.masked_equal(lightning_patch_data, -10000.0)

        if lightning_total_sum is None:
            num_lightning_channels = lightning_patch_data.shape[1]
            lightning_total_sum = np.zeros(num_lightning_channels, dtype=np.float64)
            lightning_total_sum_sq = np.zeros(num_lightning_channels, dtype=np.float64)
            lightning_total_count = np.zeros(num_lightning_channels, dtype=np.int64)

        lightning_total_sum += lightning_masked_patch_data.sum(axis=(0, 2, 3)).filled(0)
        lightning_total_sum_sq += np.ma.power(lightning_masked_patch_data, 2).sum(axis=(0, 2, 3)).filled(0)
        lightning_total_count += lightning_masked_patch_data.count(axis=(0, 2, 3))

    # Calculate mean and std for sat
    assert sat_total_sum is not None
    sat_mean = np.divide(sat_total_sum, sat_total_count, out=np.zeros_like(sat_total_sum), where=sat_total_count != 0)
    sat_variance = np.divide(sat_total_sum_sq, sat_total_count, out=np.zeros_like(sat_total_sum_sq), where=sat_total_count != 0) - np.square(sat_mean)
    sat_std = np.sqrt(sat_variance)

    # Calculate mean and std for lightning
    assert lightning_total_sum is not None
    lightning_mean = np.divide(lightning_total_sum, lightning_total_count, out=np.zeros_like(lightning_total_sum), where=lightning_total_count != 0)
    lightning_variance = np.divide(lightning_total_sum_sq, lightning_total_count, out=np.zeros_like(lightning_total_sum_sq), where=lightning_total_count != 0) - np.square(lightning_mean)
    lightning_std = np.sqrt(lightning_variance)

    print("\nSat Mean for each channel:")
    print(sat_mean)
    print("\nSat Standard deviation for each channel:")
    print(sat_std)

    print("\nLightning Mean for each channel:")
    print(lightning_mean)
    print("\nLightning Standard deviation for each channel:")
    print(lightning_std)

if __name__ == "__main__":
    compute_mean_std()
