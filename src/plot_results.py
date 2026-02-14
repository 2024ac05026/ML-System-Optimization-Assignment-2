import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_all():
    df = pd.read_csv("results/experiments.csv")

    os.makedirs("results/plots", exist_ok=True)

    # -----------------------------
    # 1. Speedup Plot
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(df["P"], df["speedup"], marker="o", label="Observed Speedup")
    plt.plot(df["P"], df["P"], linestyle="--", label="Ideal Speedup (Linear)")
    plt.xlabel("Number of Workers (P)")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Workers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/speedup.jpg", dpi=300)
    plt.close()

    # -----------------------------
    # 2. Efficiency Plot
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(df["P"], df["efficiency_percent"], marker="o")
    plt.xlabel("Number of Workers (P)")
    plt.ylabel("Efficiency (%)")
    plt.title("Efficiency vs Workers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/efficiency.jpg", dpi=300)
    plt.close()

    # -----------------------------
    # 3. Training Time Bar Chart
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.bar(df["P"].astype(str), df["mean_time"])
    plt.xlabel("Number of Workers (P)")
    plt.ylabel("Mean Training Time (seconds)")
    plt.title("Training Time vs Workers")
    plt.tight_layout()
    plt.savefig("results/plots/training_time.jpg", dpi=300)
    plt.close()

    # -----------------------------
    # 4. Communication Overhead Estimate
    # -----------------------------
    ideal_time = df["mean_time"].iloc[0] / df["P"]
    comm_overhead = df["mean_time"] - ideal_time

    plt.figure(figsize=(8, 6))
    plt.plot(df["P"], comm_overhead, marker="o")
    plt.xlabel("Number of Workers (P)")
    plt.ylabel("Estimated Overhead (seconds)")
    plt.title("Estimated Communication & Overhead vs Workers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/plots/overhead.jpg", dpi=300)
    plt.close()

    print("All plots saved to results/plots/")


if __name__ == "__main__":
    plot_all()
