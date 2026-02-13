import os
import time
import pandas as pd
import numpy as np

from src.config import ExperimentConfig
from src.data_loader import DataLoader
from src.distributed_trainer import DistributedXGBoostTrainer


def run_single_experiment(P: int, config: ExperimentConfig, show_dashboard=False):

    trainer = DistributedXGBoostTrainer(n_workers=P, config=config)
    trainer.setup_cluster(show_dashboard=show_dashboard)

    model, training_time = trainer.train(config.X_train, config.y_train)
    metrics = trainer.evaluate(model, config.X_test, config.y_test)

    if show_dashboard:
        print(f"Dask dashboard available at: {trainer.client.dashboard_link}")
        input("Open the dashboard now. Press Enter to continue with training...")

    trainer.cleanup()

    return training_time, metrics["accuracy"], metrics["auc"]


def main():

    config = ExperimentConfig()

    # Load real dataset
    loader = DataLoader(config)
    X, y = loader.load_dataset()
    X, y = loader.preprocess(X, y)
    X_train, X_test, y_train, y_test = loader.split_data(X, y)

    # Attach to config for trainer access
    config.X_train = X_train
    config.X_test = X_test
    config.y_train = y_train
    config.y_test = y_test

    results = []

    baseline_time = None

    for P in config.WORKER_COUNTS:

        run_times = []
        accuracies = []
        aucs = []

        print(f"\nRunning P={P}")

        for run_id in range(config.N_RUNS_PER_CONFIG):

            print(f"  Run {run_id+1}/{config.N_RUNS_PER_CONFIG}")

            training_time, accuracy, auc = run_single_experiment(P, config, show_dashboard=False)

            run_times.append(training_time)
            accuracies.append(accuracy)
            aucs.append(auc)

        mean_time = np.mean(run_times)
        std_time = np.std(run_times)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        if P == 1:
            baseline_time = mean_time
            speedup = 1.0
        else:
            speedup = baseline_time / mean_time

        efficiency = speedup / P * 100

        results.append({
            "P": P,
            "mean_time": mean_time,
            "std_time": std_time,
            "speedup": speedup,
            "efficiency_percent": efficiency,
            "mean_auc": mean_auc,
            "std_auc": std_auc
        })

    df = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/experiments.csv", index=False)

    print("\nExperiments completed.")
    print(df)


if __name__ == "__main__":
    main()
