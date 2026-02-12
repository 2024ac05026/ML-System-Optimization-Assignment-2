# run_experiments.py

import os
import json
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.distributed_trainer import DistributedXGBoostTrainer
from src.plot_results import plot_speedup


# ---------------------------------------------
# Simple Experiment Config Class (If not exists)
# ---------------------------------------------
class ExperimentConfig:
    def __init__(self):
        self.n_estimators = 50
        self.max_depth = 6
        self.learning_rate = 0.1


# ---------------------------------------------
# Generate Synthetic Dataset
# ---------------------------------------------
def load_data():

    X, y = make_classification(
        n_samples=20000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------
# Main Experiment Loop
# ---------------------------------------------
def main():

    X_train, X_test, y_train, y_test = load_data()

    config = ExperimentConfig()
    worker_list = [1, 2, 4, 8]

    results = {}
    baseline_time = None

    for P in worker_list:

        print(f"\nRunning experiment with {P} workers")

        trainer = DistributedXGBoostTrainer(
            n_workers=P,
            config=config
        )

        trainer.setup_cluster()

        model, training_time = trainer.train(X_train, y_train)

        metrics = trainer.evaluate(model, X_test, y_test)

        trainer.cleanup()

        if P == 1:
            baseline_time = training_time
            speedup = 1.0
        else:
            speedup = baseline_time / training_time

        results[P] = {
            "training_time": training_time,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "speedup": speedup
        }

        print(f"P={P}")
        print(f"Training Time: {training_time:.4f}s")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Speedup: {speedup:.2f}")

    # ---------------------------------------------
    # Save Results
    # ---------------------------------------------
    os.makedirs("results", exist_ok=True)

    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    df = pd.DataFrame(results).T
    df.to_csv("results/experiment_results.csv")

    print("\nExperiments completed. Results saved in results/ folder.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
    plot_speedup()
