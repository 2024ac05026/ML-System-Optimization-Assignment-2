# src/data_loader.py (interface for Person 3)
class DataLoader:
    def __init__(self, config):
        self.config = config
    
    def load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load Higgs dataset"""
        pass
    
    def preprocess(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Handle missing values, normalize if needed"""
        pass
    
    def split_data(self, X, y) -> Tuple[...]:
        """Train/test split"""
        pass
    
    def partition_for_dask(self, X, y, n_workers) -> Tuple[dask.array, dask.array]:
        """Convert to Dask arrays with proper chunking"""
        pass