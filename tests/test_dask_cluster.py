from src.dask_cluster import setup_dask_cluster, shutdown_dask_cluster

def test_dask_cluster_startup():
    cluster, client = setup_dask_cluster(n_workers=2)

    info = client.scheduler_info()
    assert len(info["workers"]) == 2

    shutdown_dask_cluster(cluster, client)
