# src/distributed_trainer.py

import time
from typing import Tuple, Dict

import xgboost as xgb
from dask.distributed import Client, LocalCluster
from dask import array as da
from sklearn.metrics import accuracy_score, roc_auc_score


class DistributedXGBoostTrainer:
    def __init__(self, n_workers: int, config):
        self.n_workers = n_workers
        self.config = config
        self.cluster = None
        self.client = None

    # ------------------------------------------------
    # Setup Dask LocalCluster
    # ------------------------------------------------
    def setup_cluster(self):
        self.cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=1
        )
        self.client = Client(self.cluster)

    # ------------------------------------------------
    # Train Distributed XGBoost
    # ------------------------------------------------
    def train(self, X, y) -> Tuple[xgb.Booster, float]:

        # Convert to Dask arrays
        X_dask = da.from_array(X, chunks=(len(X)//self.n_workers, X.shape[1]))
        y_dask = da.from_array(y, chunks=(len(y)//self.n_workers,))

        dtrain = xgb.dask.DaskDMatrix(self.client, X_dask, y_dask)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": self.config.max_depth,
            "eta": self.config.learning_rate,
        }

        start_time = time.time()

        output = xgb.dask.train(
            self.client,
            params,
            dtrain,
            num_boost_round=self.config.n_estimators,
        )

        end_time = time.time()
        training_time = end_time - start_time

        model = output["booster"]

        return model, training_time

    # ------------------------------------------------
    # Evaluate Model
    # ------------------------------------------------
    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:

        X_dask = da.from_array(X_test, chunks=(len(X_test)//self.n_workers, X_test.shape[1]))
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
    # Cleanup Cluster
    # ------------------------------------------------
    def cleanup(self):
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()
