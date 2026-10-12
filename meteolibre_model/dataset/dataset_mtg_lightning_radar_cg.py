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
import pyarrow.parquet as pq

import torch
import torch.distributed as dist


from suncalc import get_position


class MeteoLibreMapDataset(torch.utils.data.Dataset):
    """
    Map-style Dataset with LRU-cached parquet reads over the local clone of
    the meteolibre-dev/weather_mtg_europe_lightning_128_1km dataset.

    The order of Parquet files is shuffled ONCE in __init__ (single seeded
    generator, offset by the DDP rank if distributed is initialised) and is
    then shared by all forked DataLoader workers: every worker uses the SAME
    index -> file mapping.

    This is critical for the per-worker LRU cache: as workers advance through
    the shared shuffled file order, the working set at any moment is only a
    handful of files, so whichever worker grabs a batch, the file is already
    hot in its cache. (The previous design reshuffled files per worker, which
    made every batch a likely cold miss and blew up RAM with up to
    num_workers * cache_size DISTINCT DataFrames -- workers stalled, GPU 0%.)

    It also guarantees full per-epoch coverage: with shuffle=False the
    DataLoader fetches every index exactly once and the fixed mapping means
    every record is seen exactly once per epoch.

    Args:
        localrepo (str): Root of the local dataset clone. Parquet files are
            read from *every* ``{localrepo}/data*/`` subdirectory (data/,
            data_v1/, data_v2/, data_wa/, ...), so new folders are picked up
            automatically.
        cache_size (int): The number of DataFrames to keep in the in-memory cache.
        seed (int): Base seed for the one-time file-order shuffle (offset by
                    the DDP rank under distributed training, so ranks explore
                    different orders).
    """

    def __init__(self, localrepo: str, cache_size: int = 8, seed: int = 42, nb_temporal: int = 5):
        super().__init__()
        self.localrepo = localrepo
        self.cache_size = cache_size
        self.seed = seed  # Store base seed
        self.nb_temporal = nb_temporal

        # Discover parquet files from every data*/ subdirectory (data/,
        # data_v1/, data_v2/, data_wa/, ... -- new folders are picked up
        # automatically), sorted for a stable base order.
        data_dirs = sorted(
            d
            for d in glob.glob(os.path.join(self.localrepo, "data*"))
            if os.path.isdir(d)
        )
        candidates = sorted(
            pq_file
            for d in data_dirs
            for pq_file in glob.glob(os.path.join(d, "*.parquet"))
        )

        if not candidates:
            raise FileNotFoundError(
                f"No Parquet files found under any 'data*/' subdirectory of "
                f"'{self.localrepo}'. Found dirs: {data_dirs or '<none>'}"
            )

        # Shuffle ONCE here so forked DataLoader workers all inherit the SAME
        # index -> file mapping (cache locality + full per-epoch coverage).
        # Offset by the DDP rank so distributed ranks explore different orders.
        shuffle_seed = self.seed
        if dist.is_available() and dist.is_initialized():
            shuffle_seed += dist.get_rank()
        g = torch.Generator()
        g.manual_seed(shuffle_seed)
        perm = torch.randperm(len(candidates), generator=g).tolist()
        self.base_file_paths = [candidates[i] for i in perm]
        self.file_paths = list(self.base_file_paths)

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

        radar_patch_data = (
            np.frombuffer(record["radar_data"], dtype=record["radar_dtype"])
            .reshape(record["radar_shape"])
            .copy()
        )

        isnan_radar = np.isnan(radar_patch_data)
        radar_patch_data = np.where(isnan_radar, -10000.0, radar_patch_data)

        # now concat :)
        sat_patch_data = np.concatenate([sat_patch_data, groundheight_patch_data, radar_patch_data], axis=1)

        cg_lightning_patch_data = (
            np.frombuffer(record["cg_lightning_data"], dtype=record["cg_lightning_dtype"])
            .reshape(record["cg_lightning_shape"])
            .copy()
        )
        # cg_lightning is all-NaN in data_v2/ and data_wa/ (CG product
        # unavailable for those regions/times): a NaN here would flow
        # straight into the loss -> fill with 0.0 (= no CG strikes).
        cg_lightning_patch_data = np.where(
            np.isnan(cg_lightning_patch_data), 0.0, cg_lightning_patch_data
        )

        lightning_patch_data = (
            np.frombuffer(record["lightning_data"], dtype=record["lightning_dtype"])
            .reshape(record["lightning_shape"])
            .copy()
        )
        # defensive: same treatment for the total-lightning channel (currently
        # clean in every folder, but cheap insurance against NaN loss).
        lightning_patch_data = np.where(
            np.isnan(lightning_patch_data), 0.0, lightning_patch_data
        )

        # in case of bad formatting
        if record["sat_shape"][0] > self.nb_temporal:
            sat_patch_data = sat_patch_data[:self.nb_temporal, :, :, :]

        if record["lightning_shape"][0] > self.nb_temporal:
            lightning_patch_data = lightning_patch_data[:self.nb_temporal, :, :, :]

        if record["cg_lightning_shape"][0] > self.nb_temporal:
            cg_lightning_patch_data = cg_lightning_patch_data[:self.nb_temporal, :, :, :]

        lightning_cc_cg = np.concatenate([lightning_patch_data, cg_lightning_patch_data], axis=1)

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
            "lightning_patch_data": torch.from_numpy(lightning_cc_cg),
            "spatial_position": torch.tensor(
                [result["azimuth"], result["altitude"], result_noon["altitude"], lat / 10.0]
            ),
        }

        return batch_dict

    def __getitem__(self, index: int) -> dict:
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

        # Skip a corrupt/unreadable row by wrapping to a neighbour
        # (live handler -- wraps the actual fetch, unlike the old dead code).
        try:
            record = data_df.iloc[row_index_in_file]
            return self._preprocess(date, record)
        except Exception as e:
            print(
                f"[MeteoLibreMapDataset] bad row index={index}, "
                f"file_index={file_index}, row={row_index_in_file}. Error: {e}",
                flush=True,
            )
            return self.__getitem__((index + 1) % self.total_records)

