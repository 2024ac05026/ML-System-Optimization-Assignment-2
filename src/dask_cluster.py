from dask.distributed import Client, LocalCluster
from typing import Tuple
import logging
import os

logger = logging.getLogger(__name__)


def setup_dask_cluster(
    n_workers: int,
    threads_per_worker: int = 1,
    memory_limit: str = "8GB",
    dashboard: bool = False,
) -> Tuple[LocalCluster, Client]:
    """
    Initialize a Dask LocalCluster.

    Dashboard is optional and should be enabled only for debugging.
    """

    dashboard_address = ":8787" if dashboard else None

    logger.info(
        f"Starting Dask cluster | workers={n_workers}, "
        f"threads/worker={threads_per_worker}, memory={memory_limit}, "
        f"dashboard={dashboard}"
    )

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,              # REQUIRED: avoid GIL
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        silence_logs=logging.ERROR,
    )

    client = Client(cluster)
    client.wait_for_workers(n_workers)

    if dashboard:
        logger.info(f"Dask dashboard available at: {client.dashboard_link}")

    return cluster, client


def shutdown_dask_cluster(cluster: LocalCluster, client: Client):
    logger.info("Shutting down Dask cluster")
    client.close()
    cluster.close()
