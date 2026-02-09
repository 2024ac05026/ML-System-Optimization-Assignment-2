from src.dask_cluster import setup_dask_cluster, shutdown_dask_cluster
import os

def get_pid():
    return os.getpid()

def main():
    cluster, client = setup_dask_cluster(n_workers=4, dashboard=True)

    pids = client.run(get_pid)
    unique_pids = set(pids.values())

    print("Worker PIDs:", unique_pids)
    print("Unique processes:", len(unique_pids))

    assert len(unique_pids) == 4

    print("Dashboard:", client.dashboard_link)
    input("Open the dashboard now. Press Enter to exit.")

    shutdown_dask_cluster(cluster, client)

if __name__ == "__main__":
    main()
