# src/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt


def plot_speedup(csv_path="results/experiment_results.csv"):

    df = pd.read_csv(csv_path, index_col=0)

    P_values = df.index.astype(int)
    speedup = df["speedup"]

    plt.figure()
    plt.plot(P_values, speedup, marker='o')
    plt.xlabel("Number of Workers (P)")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Number of Workers")
    plt.grid(True)
    plt.savefig("results/speedup_plot.png")
    plt.close()

    print("Speedup plot saved to results/speedup_plot.png")
