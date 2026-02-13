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


HIGGS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
)


class DataLoader:
    def __init__(self, config):
        self.config = config

    # ------------------------------------------------
    # Ensure Dataset Exists (Download if Missing)
    # ------------------------------------------------
    def _ensure_dataset_exists(self):
        dataset_path = self.config.DATASET_PATH
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        gz_path = dataset_path + ".gz"


        if os.path.exists(dataset_path):
            print("Higgs dataset found locally.")
            return
        
        if os.path.exists(gz_path):
            print("Compressed Higgs dataset found locally. Extracting...")
            # Extract
            with gzip.open(gz_path, "rb") as f_in:
                with open(dataset_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return

        

        print("Higgs dataset not found. Downloading 2.6 GB file...")
        print("You could also go to " + HIGGS_URL + " and download it manually if you prefer. Place it inside data folder and name it higgs.csv.gz")

        # Download compressed file
        urllib.request.urlretrieve(HIGGS_URL, gz_path)
        # Extract
        with gzip.open(gz_path, "rb") as f_in:
            with open(dataset_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print("Download complete.")

    # ------------------------------------------------
    # Load Dataset
    # ------------------------------------------------
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Higgs dataset.

        Format:
        Column 0 = label
        Columns 1-28 = features
        """

        self._ensure_dataset_exists()

        print("Loading dataset...")

        df = pd.read_csv(
            self.config.DATASET_PATH,
            header=None,
            dtype=np.float32,
            nrows=self.config.N_SAMPLES
        )

        y = df.iloc[:, 0].values.astype(np.int32, copy=False)
        X = df.iloc[:, 1:].values  # already float32

        print(f"Dataset loaded: {X.shape}")

        return X, y

    # ------------------------------------------------
    # Preprocess
    # ------------------------------------------------
    def preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Minimal preprocessing:
        - Ensure contiguous arrays
        - Ensure correct dtype
        """

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int32)

        return X, y

    # ------------------------------------------------
    # Train/Test Split (Stratified)
    # ------------------------------------------------
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        return train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=42,
            stratify=y
        )

    # ------------------------------------------------
    # Partition for Dask (Horizontal Split)
    # ------------------------------------------------
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

        # Even chunk distribution
        base_chunk = n_samples // n_workers
        remainder = n_samples % n_workers

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

        print(
            f"Dask partitioned into {n_workers} chunks "
            f"(~{base_chunk} rows per worker)"
        )

        return X_dask, y_dask
