# src/data_loader.py

from typing import Tuple
import pandas as pd
import numpy as np
import dask.array as da
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config):
        self.config = config

    # ------------------------------------------------
    # Load Higgs Dataset
    # ------------------------------------------------
    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load Higgs dataset from CSV.

        Expected format:
        First column = label
        Remaining columns = features
        """

        df = pd.read_csv(
            self.config.DATASET_PATH,
            header=None
        )

        # Higgs format:
        # Column 0 = label
        # Columns 1-28 = features
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]

        return X, y

    # ------------------------------------------------
    # Preprocess Dataset
    # ------------------------------------------------
    def preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocessing:
        - Convert to float32 (memory efficient)
        - Handle missing values (Higgs typically has none)
        - No normalization needed for tree-based models
        """

        # Fill any potential NaNs
        X = X.fillna(0)

        # Convert to numpy float32 (critical for memory)
        X_np = X.values.astype(np.float32)
        y_np = y.values.astype(np.float32)

        return X_np, y_np

    # ------------------------------------------------
    # Train/Test Split
    # ------------------------------------------------
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        80/20 train-test split
        Stratified for binary classification
        """

        return train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=42,
            stratify=y
        )

    # ------------------------------------------------
    # Partition for Dask
    # ------------------------------------------------
    def partition_for_dask(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_workers: int
    ) -> Tuple[da.Array, da.Array]:
        """
        Convert numpy arrays into Dask arrays.

        Chunking strategy:
        - Horizontal partitioning (data parallelism)
        - ~N/P rows per worker
        """

        import math

        chunk_size = math.ceil(len(X) / n_workers)

        X_dask = da.from_array(
            X,
            chunks=(chunk_size, X.shape[1])
        )

        y_dask = da.from_array(
            y,
            chunks=(chunk_size,)
        )

        return X_dask, y_dask
