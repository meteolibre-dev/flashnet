"""
This is the module to handle the dataloading from the huggingface dataset

This is the dataloader for the meteolibre-dev/weather_mtg_europe_lightning_128_1km huggingface dataset
"""


from datetime import datetime
import pandas as pd
import os
import glob
from collections import OrderedDict
import bisect
import ast

import numpy as np
import scipy.sparse as sp
import random
import pyarrow.parquet as pq

from torch.utils.data import get_worker_info
import torch
import torch.distributed as dist


from suncalc import get_position


class MeteoLibreMapDataset(torch.utils.data.Dataset):
    """
    An advanced map-style Dataset that improves upon the cached version by
    shuffling the order of Parquet files differently for each DataLoader worker.

    This ensures that workers are likely to request different files at the
    same time, which can improve data loading randomness and potentially reduce
    contention for the same files if they were on a shared network drive.

    Args:
        localrepo (str): The path to the local repository.
        cache_size (int): The number of DataFrames to keep in the in-memory cache.
        seed (int): A base seed used to ensure reproducible shuffling across runs.
                    Each worker's shuffle seed will be `seed + worker_id`.
    """

    def __init__(self, localrepo: str, cache_size: int = 8, seed: int = 42, nb_temporal: int = 5):
        super().__init__()
        self.localrepo = localrepo
        self.cache_size = cache_size
        self.seed = seed  # Store base seed
        self.nb_temporal = nb_temporal

        # Find all parquet files. We start with a sorted list for consistency.
        data_path = os.path.join(self.localrepo, "data", "*.parquet")
        self.base_file_paths = sorted(glob.glob(data_path))

        # we remove the last one
        self.base_file_paths = self.base_file_paths

        if not self.base_file_paths:
            raise FileNotFoundError(f"No Parquet files found at '{data_path}'.")

        self.file_paths = self.base_file_paths

        # Initialize an LRU cache for this worker
        self.cache = OrderedDict()

        # Calculate the total number of records by inspecting each file
        self.records_per_file_list = [
            sum(p.count_rows() for p in pq.ParquetDataset(fp).fragments)
            for fp in self.base_file_paths
        ]
        self.total_records = sum(self.records_per_file_list)

        # Create a cumulative sum of records to quickly find the file for an index
        self.cumulative_records = np.cumsum(
            [0] + self.records_per_file_list[:-1]
        ).tolist()

    def __len__(self) -> int:
        return self.total_records

    def _get_dataframe(self, file_index: int) -> pd.DataFrame:
        if file_index in self.cache:
            self.cache.move_to_end(file_index)
            return self.cache[file_index]

        file_path = self.file_paths[file_index]
        data_df = pd.read_parquet(file_path)
        self.cache[file_index] = data_df

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return data_df

    def _preprocess(self, date, record: pd.Series) -> dict:
        # satellite data
        sat_patch_data = (
            np.frombuffer(record["sat_data"], dtype=record["sat_dtype"])
            .reshape(record["sat_shape"])
            .copy()
        )

        groundheight_patch_data = (
            np.frombuffer(record["elevation_data"], dtype=record["elevation_dtype"])
            .reshape(record["elevation_shape"])
            .copy()
        )

        groundheight_patch_data = np.where(groundheight_patch_data < 0, -100, groundheight_patch_data)
        groundheight_patch_data = groundheight_patch_data[None, None, :, :].repeat(self.nb_temporal, axis=0)

        # now concat
        sat_patch_data = np.concatenate([sat_patch_data, groundheight_patch_data], axis=1)

        lightning_patch_data = (
            np.frombuffer(record["lightning_data"], dtype=record["lightning_dtype"])
            .reshape(record["lightning_shape"])
            .copy()
        )

        # in case of bad formatting
        if record["sat_shape"][0] > self.nb_temporal:
            sat_patch_data = sat_patch_data[:self.nb_temporal, :, :, :]

        if record["lightning_shape"][0] > self.nb_temporal:
            lightning_patch_data = lightning_patch_data[:self.nb_temporal, :, :, :]

        # now we try to retrieve the longitude latitude of the patch to get the sun orientation on the patches
        long = record["lon"] # longitude
        lat = record["lat"] # latitude

        result = get_position(date, long, lat)

        # here we want to geth information about the sun top altitude of the day 
        # (to get an idea of what type of the year we are)
        date_12 = date.replace(hour=12, minute=0, second=0, microsecond=0)
        result_noon = get_position(date_12, long, lat)

        batch_dict = {
            "sat_patch_data": torch.from_numpy(sat_patch_data),
            "lightning_patch_data": torch.from_numpy(lightning_patch_data),
            "spatial_position": torch.tensor(
                [result["azimuth"], result["altitude"], result_noon["altitude"], lat / 10.0]
            ),
        }



        return batch_dict

    def __getitem__(self, index: int) -> dict:
        if not getattr(self, "worker_initialized", False):
            worker_info = get_worker_info()
            if worker_info is not None:
                # We are in a worker process. Shuffle the files based on worker ID.
                worker_id = worker_info.id
                # print("worker_id", worker_id)

                # Get rank of the process to ensure different shuffling across GPUs
                rank = 0
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()

                # print("rank", rank)

                # Create a generator with a seed unique to this worker and the base seed
                g = torch.Generator()

                seed = (
                    self.seed
                    + worker_id
                    + rank * worker_info.num_workers
                    + random.randint(0, 1000)
                )  # + round(datetime.now().timestamp() * 1000)
                g.manual_seed(seed)

                # Get a random permutation of indices and apply it to the file list
                perm = torch.randperm(len(self.base_file_paths), generator=g).tolist()
                self.file_paths = [self.base_file_paths[i] for i in perm]

                # recompute
                self.records_per_file_list = [self.records_per_file_list[i] for i in perm]
                self.cumulative_records = np.cumsum([0] + self.records_per_file_list[:-1]).tolist()

            self.worker_initialized = True

        if index < 0 or index >= self.total_records:
            raise IndexError(
                f"Index {index} out of range for dataset with size {self.total_records}"
            )

        file_index = bisect.bisect_right(self.cumulative_records, index) - 1
        row_index_in_file = index - self.cumulative_records[file_index]

        # get date from file name
        file_path = self.file_paths[file_index]
        filename = os.path.basename(file_path)
        date_str = filename.split("_")[:2]
        date_str = "_".join(date_str)

        date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")

        data_df = self._get_dataframe(file_index)

        record = data_df.iloc[row_index_in_file]
        return self._preprocess(date, record)

        try:
            record = data_df.iloc[row_index_in_file]
            return self._preprocess(date, record)
        except Exception as e:
            print(
                f"Bad indexing for index {index}, file_index {file_index}, row_index {row_index_in_file}. Error: {e}"
            )
            return self.__getitem__((index + 64) % self.total_records)

