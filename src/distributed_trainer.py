import time
from typing import Tuple, Dict

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import xgboost as xgb
from dask import array as da
from sklearn.metrics import accuracy_score, roc_auc_score

from src.dask_cluster import setup_dask_cluster, shutdown_dask_cluster


class DistributedXGBoostTrainer:
    def __init__(self, n_workers: int, config):
        self.n_workers = n_workers
        self.config = config
        self.cluster = None
        self.client = None

    # ------------------------------------------------
    # Setup Dask Cluster
    # ------------------------------------------------
    def setup_cluster(self, show_dashboard=False):
        self.cluster, self.client = setup_dask_cluster(
            n_workers=self.n_workers,
            threads_per_worker=1,
            memory_limit="8GB",
            dashboard=show_dashboard
        )

    # ------------------------------------------------
    # Train Distributed XGBoost
    # ------------------------------------------------
    def train(self, X, y) -> Tuple[xgb.Booster, float]:

        if self.client is None:
            raise RuntimeError("Dask cluster not initialized. Call setup_cluster() first.")

        # Chunk size per worker
        chunk_size = len(X) // self.n_workers

        X_dask = da.from_array(X, chunks=(chunk_size, X.shape[1]))
        y_dask = da.from_array(y, chunks=(chunk_size,))

        dtrain = xgb.dask.DaskDMatrix(self.client, X_dask, y_dask)

        params = self.config.XGBOOST_PARAMS

        start_time = time.time()

        output = xgb.dask.train(
            self.client,
            params,
            dtrain,
            num_boost_round=self.config.N_ESTIMATORS,
        )

        training_time = time.time() - start_time

        model = output["booster"]

        return model, training_time

    # ------------------------------------------------
    # Evaluate Model
    # ------------------------------------------------
    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:

        if self.client is None:
            raise RuntimeError("Dask cluster not initialized.")

        chunk_size = len(X_test) // self.n_workers

        X_dask = da.from_array(X_test, chunks=(chunk_size, X_test.shape[1]))
        dtest = xgb.dask.DaskDMatrix(self.client, X_dask)

        preds = xgb.dask.predict(self.client, model, dtest)
        preds = preds.compute()

        preds_binary = (preds > 0.5).astype(int)

        accuracy = accuracy_score(y_test, preds_binary)
        auc = roc_auc_score(y_test, preds)

        return {
            "accuracy": accuracy,
            "auc": auc
        }

    # ------------------------------------------------
    # Cleanup Cluster (Uses infra layer)
    # ------------------------------------------------
    def cleanup(self):
        if self.cluster and self.client:
            shutdown_dask_cluster(self.cluster, self.client)
