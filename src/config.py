# src/config.py (YOU create this)
class ExperimentConfig:
    # Dataset
    DATASET_PATH = "data/higgs.csv"
    N_SAMPLES = 11_000_000
    N_FEATURES = 28
    TEST_SIZE = 0.2
    
    # XGBoost params (MUST be constant across all experiments)
    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100,
        'random_state': 42,
        'eval_metric': 'auc'
    }
    
    # Experiment settings
    WORKER_COUNTS = [1, 2, 4, 8]
    N_RUNS_PER_CONFIG = 3  # For averaging
    
    # Output
    RESULTS_FILE = "results/experiments.csv"
    PLOTS_DIR = "results/plots/"