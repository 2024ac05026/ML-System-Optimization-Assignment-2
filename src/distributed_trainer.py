# src/distributed_trainer.py (interface for Person 4)
class DistributedXGBoostTrainer:
    def __init__(self, n_workers: int, config: ExperimentConfig):
        pass
    
    def setup_cluster(self):
        """Initialize Dask LocalCluster"""
        pass
    
    def train(self, X, y) -> Tuple[xgb.Booster, float]:
        """
        Train and return (model, training_time)
        """
        pass
    
    def evaluate(self, model, X_test, y_test) -> Dict[str, float]:
        """Return {'auc': ..., 'accuracy': ...}"""
        pass
    
    def cleanup(self):
        """Close cluster"""
        pass