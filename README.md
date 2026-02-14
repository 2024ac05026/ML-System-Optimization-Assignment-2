# ML System Optimization — Assignment 2

## Project Abstract

This project evaluates **system-level optimization** techniques for gradient-boosted decision trees (XGBoost) on a large tabular dataset (HIGGS). We quantify how **parallel worker counts** (1, 2, 4, 8) impact end‑to‑end training **time**, **throughput**, and **AUC**, while holding model hyperparameters constant across experiments. Each configuration is run multiple times to reduce variance, and consolidated results are saved for analysis and visualization.

---

## Table of Contents

- Project Abstract
- Repository Structure
- Environment Setup
- Dataset
- Configuration
- How-to-run
- Outputs
- Results Interpretation
- Troubleshooting

---

## Repository Structure

ML-System-Optimization-Assignment-2/
├─ src/
│ ├─ **init**.py
│ ├─ experiment_runner.py # Orchestrates experiments with varying worker counts
│ ├─ config.py # Central configuration (dataset path, XGBoost params, runs)
│ ├─ utils.py # (Optional) timing, logging, plotting helpers
│ └─ ...
├─ data/
│ └─ HIGGS.csv # Input dataset (not tracked in git, large)
├─ results/
│ ├─ experiments.csv # Aggregated results (appended/overwritten by runs)
│ └─ plots/ # Saved figures (speedup, efficiency, AUC, etc.)
├─ README.md
└─ requirements.txt

> If you don't see `data/HIGGS.csv`, download or place it under `data/`.

---

## Environment Setup

```bash
# 1) Create an isolated environment
python -m venv .venv

# 2) Activate it
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# If you don't have requirements.txt yet, minimally install:
# pip install xgboost pandas numpy scikit-learn matplotlib seaborn tqdm
```

**(Optional) Verify CPU/threads**

```bash
python - << 'PY'
import os, multiprocessing
print('Logical CPUs:', os.cpu_count())
print('Multiprocessing CPUs:', multiprocessing.cpu_count())
PY
```

---

## Dataset

- **Path**: `data/HIGGS.csv`
- **Samples used**: 5,000,000 (`N_SAMPLES` in config)
- **Features**: 28 (tabular)
- **Split**: 80/20 train/test

> Ensure you have adequate disk and RAM. Five million rows with 28 features can be memory‑intensive depending on dtype and loader.

---

## Configuration

Core experiment parameters live in `src/config.py`:

```python
# src/config.py
class ExperimentConfig:
    # Dataset
    DATASET_PATH = "data/HIGGS.csv"
    # N_SAMPLES = 11_000_000
    N_SAMPLES = 5_000_000
    N_FEATURES = 28
    TEST_SIZE = 0.2

    # XGBoost params (MUST be constant across all experiments)
    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'auc'
    }
    N_ESTIMATORS = 500

    # Experiment settings
    WORKER_COUNTS = [1, 2, 4, 8]
    N_RUNS_PER_CONFIG = 3  # For averaging

    # Output
    RESULTS_FILE = "results/experiments.csv"
    PLOTS_DIR = "results/plots/"
```

> Keep **XGBOOST_PARAMS** and **N_ESTIMATORS** constant across all runs so that timing differences reflect **system parallelism**, not model complexity changes.

---

## How to Run

The entire experiment sweep is launched via the module entry point:

```bash
python -m src.experiment_runner
```

**What it does** (typical flow):

1. Loads `ExperimentConfig`.
2. Reads `data/HIGGS.csv` and selects `N_SAMPLES` and `N_FEATURES`.
3. Splits data into train/test with `TEST_SIZE`.
4. For each `workers` in `[1, 2, 4, 8]`:
   - Runs training **`N_RUNS_PER_CONFIG`** times to reduce variance.
   - Records wall‑clock time, AUC (or other metrics), and resource stats.
5. Appends/writes a consolidated rowset to `results/experiments.csv`.
6. Generates plots under `results/plots/`.

**Notes**

- Depending on how parallelism is implemented, the code may set XGBoost’s `nthread`, a joblib/sklearn `n_jobs`, or python multiprocessing pools internally. Always keep the **model hyperparameters fixed** while varying only the **parallel workers**.

---

## Outputs

- **Tabular results** → `results/experiments.csv` (one row per run + aggregates)
- **Plots** → `results/plots/`
  - `speedup_vs_workers.png`
  - `efficiency_vs_workers.png`
  - `training_time_distribution.png`
  - `auc_vs_workers.png`

_(Exact filenames may differ; this README matches the configuration you shared and common naming.)_

---

## Results Interpretation

- **Speedup**: `T1 / Tp`, where `T1` is mean time with 1 worker and `Tp` with `p` workers.
- **Parallel Efficiency**: `Speedup / p`. Expect diminishing returns as p increases due to overheads (I/O, synchronization, memory bandwidth).
- **AUC stability**: Should remain roughly constant across worker counts since hyperparameters and data are the same; large deviations may indicate nondeterminism or data sharding issues.

---

## Troubleshooting

- **File not found**: Ensure `data/HIGGS.csv` exists at `ExperimentConfig.DATASET_PATH`.
- **Out of memory**: Reduce `N_SAMPLES`, or use chunked loading/feather/parquet. Verify that `tree_method='hist'` is set (more memory‑efficient).
- **Slow disk I/O**: Place dataset on SSD; avoid network mounts for first runs.
- **CPU underutilization**: Confirm the code is actually passing the `workers` value to the parallel backend (e.g., XGBoost `nthread`, joblib `n_jobs`, or multiprocessing pool size). Ensure power settings aren’t throttling cores.
