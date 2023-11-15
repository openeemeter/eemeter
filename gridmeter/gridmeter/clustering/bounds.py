from __future__ import annotations

import numpy as np

def _get_num_cluster_max_from_n_data(
    n_data: int, min_cluster_size: int, num_cluster_bound_upper: int
):
    """
    returns bounds using n_data which is number models in cluster
    """
    n_min = min_cluster_size

    min_clusters = 1
    max_clusters = num_cluster_bound_upper

    # assume we want 250 with a size of 1000
    n_set = 1000
    n_max_set = 250

    k = (n_set - n_min) * (
        np.log(
            (
                ((n_max_set - min_clusters) / (2 * max_clusters - min_clusters) + 0.5)
                ** -1
                - 1
            )
            ** -1
        )
    ) ** -1

    if not np.isfinite(k):
        """
        TODO: Figure out better way to handle this.
        Currently occurs when num_cluster_bound_upper is less than n_max_set
        """
        return min(n_data, num_cluster_bound_upper)

    num_cluster_max = (2 * max_clusters - min_clusters) * (
        1 / (1 + np.exp(-(n_data - n_min) / k)) - 0.5
    ) + min_clusters

    return int(np.floor(num_cluster_max))


def get_cluster_bounds_from_n_data(
    n_data: int,
    min_cluster_size: int,
    num_cluster_bound_lower: int,
    num_cluster_bound_upper: int,
):
    """
    function which returns lower and upper bound based off config values and number of data points
    """

    num_cluster_max = _get_num_cluster_max_from_n_data(
        n_data=n_data,
        min_cluster_size=min_cluster_size,
        num_cluster_bound_upper=num_cluster_bound_upper,
    )

    num_cluster_bounds = sorted([num_cluster_bound_lower, num_cluster_max])
    num_cluster_bounds[0] = max(num_cluster_bounds[0], 2)
    num_cluster_bounds[1] = max(num_cluster_bounds[1], 2)

    if num_cluster_bounds[0] == num_cluster_bounds[1]:
        num_cluster_bounds[1] += 1

    return num_cluster_bounds[0], num_cluster_bounds[1]


def get_cluster_bounds(
    data: np.ndarray,
    min_cluster_size: int,
    num_cluster_bound_lower: int,
    num_cluster_bound_upper: int,
):
    n_data = np.shape(data)[0]
    return get_cluster_bounds_from_n_data(
        n_data=n_data,
        min_cluster_size=min_cluster_size,
        num_cluster_bound_lower=num_cluster_bound_lower,
        num_cluster_bound_upper=num_cluster_bound_upper,
    )