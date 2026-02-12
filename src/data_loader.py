# src/data_loader.py (interface for Person 3)
# src/data_loader.py

import os
import urllib.request
import gzip
import shutil
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dask import array as da


HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"


class DataLoader:
    def __init__(self, config):
        self.config = config

    # ------------------------------------------------
    # Download dataset if not present
    # ------------------------------------------------
    def _ensure_dataset_exists(self):
        dataset_path = self.config.DATASET_PATH
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        if os.path.exists(dataset_path):
            print("Higgs dataset found locally.")
            return

        print("Higgs dataset not found. Downloading...")

        gz_path = dataset_path + ".gz"

        # Download
        urllib.request.urlretrieve(HIGGS_URL, gz_path)

        # Unzip
        with gzip.open(gz_path, "rb") as f_in:
            with open(dataset_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)

        print("Download complete.")

    # ------------------------------------------------
    # Load Dataset
    # ------------------------------------------------
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:

        self._ensure_dataset_exists()

        print("Loading dataset...")

        # No header, first column = label
        df = pd.read_csv(
            self.config.DATASET_PATH,
            header=None,
            dtype=np.float32
        )

        y = df.iloc[:, 0].values.astype(np.int32)
        X = df.iloc[:, 1:].values.astype(np.float32)

        return X, y

    # ------------------------------------------------
    # Preprocess (minimal for trees)
    # ------------------------------------------------
    def preprocess(self, X, y) -> Tuple[np.ndarray, np.ndarray]:

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)

        return X, y

    # ------------------------------------------------
    # Train/Test Split
    # ------------------------------------------------
    def split_data(self, X, y):

        return train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=42,
            shuffle=True
        )

    def partition_for_dask(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_workers: int
    ) -> Tuple[da.Array, da.Array]:

        if n_workers <= 0:
            raise ValueError("n_workers must be > 0")

        n_samples = X.shape[0]

        if n_workers > n_samples:
            raise ValueError("Number of workers exceeds number of samples.")

        # Evenly distribute remainder rows
        base_chunk = n_samples // n_workers
        remainder = n_samples % n_workers

        # Create chunk sizes list (exactly n_workers chunks)
        chunk_sizes = [
            base_chunk + 1 if i < remainder else base_chunk
            for i in range(n_workers)
        ]

        X_dask = da.from_array(
            X,
            chunks=(tuple(chunk_sizes), X.shape[1])
        )

        y_dask = da.from_array(
            y,
            chunks=(tuple(chunk_sizes),)
        )

        return X_dask, y_dask
