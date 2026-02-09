from src.dask_cluster import setup_dask_cluster, shutdown_dask_cluster

def main():
    cluster, client = setup_dask_cluster(n_workers=2)

    info = client.scheduler_info()
    print("Workers:", len(info["workers"]))
    assert len(info["workers"]) == 2

    shutdown_dask_cluster(cluster, client)
    print("Cluster shutdown OK")

if __name__ == "__main__":
    main()
