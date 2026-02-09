import time
from src.dask_cluster import setup_dask_cluster, shutdown_dask_cluster

def slow_task(x):
    time.sleep(2)
    return x

def main():
    cluster, client = setup_dask_cluster(n_workers=4, dashboard=True)

    start = time.time()
    futures = client.map(slow_task, range(4))
    client.gather(futures)
    elapsed = time.time() - start

    print(f"Elapsed time: {elapsed:.2f}s")

    print("Dashboard:", client.dashboard_link)
    input("Open the dashboard now. Press Enter to exit.")

    shutdown_dask_cluster(cluster, client)

    # If parallel: ~2s
    # If serial: ~8s
    assert elapsed < 4, "Tasks did not run in parallel!"

if __name__ == "__main__":
    main()
