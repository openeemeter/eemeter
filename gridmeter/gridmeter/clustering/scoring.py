"""
module is responsible for creating clusters

"""

from __future__ import annotations

import sys

import sklearn.metrics
import sklearn.cluster
import numpy as np

from gridmeter.clustering import const as _const


def get_max_score_from_system_size() -> float:
    """
    recreates the call to sys.float_info.max in order to
    follow what was used in ads repo.

    Making into function which executes each time
    so unforseen issues are less likely when running on
    distributed env
    """

    return sys.float_info.max**0.5


def renumber_clusters(clusters: np.ndarray, reorder: bool):
    """Takes in cluster identifiers and renumbers them.
        After merging or reclustering there are many cluster numbers left blank and need to be renumbered
            Example: [0, 1, 2, 5, 7]
        Additionally the clusters are reordered from largest cluster to smallest

    Args:
        clusters (list|np.array): an array in which a cluster number is defined for each load shape

    Returns:
        clusters (np.array): an array in which a cluster number is defined for each load shape
    """

    if reorder:
        # if outlier cluster exists, don't include it in the ordering
        uniq_id, counts = np.unique(clusters[clusters != -1], return_counts=True)
        count_order = np.argsort(counts)[::-1]

        uniq_id = uniq_id[count_order]

    else:
        uniq_id = np.unique(clusters)

    # if outlier cluster exists, don't change it
    conv = {-1: -1}
    conv.update({uniq_id[i]: i for i in range(len(uniq_id))})

    clusters = np.array([conv[idx] for idx in clusters])

    return clusters


def merge_small_clusters(clusters: np.ndarray, min_cluster_size: int):
    """
    OG DOCSTRING:
    Merges clusters which consist of less than the minumum number into the outlier cluster

    Args:
        clusters (list|np.array): A list defining what cluster each load shape belongs to
        min_cluster_size (int): Minumum number of meters for a cluster
            Options: 2 < val

    Returns:
        _type_: _description_
    """

    uniq_ids, uniq_counts = np.unique(clusters, return_counts=True)

    uniq_counts = uniq_counts[uniq_ids != -1]
    uniq_ids = uniq_ids[uniq_ids != -1]

    outlier_ids = uniq_ids[uniq_counts < min_cluster_size]
    clusters[np.isin(clusters, outlier_ids)] = -1

    return renumber_clusters(clusters, reorder=True)


def score_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    score_choice: str,
    dist_metric: str,
    min_cluster_size: int,
    cluster_bound_lower: int,
    max_non_outlier_cluster_count: int,
) -> tuple[float, bool]:
    """
    ---
    Original docstring:

    Score clusters of the given data with the selected choices.
    Small clusters are first merged to only score clusters above the minimum size
    and not in the outlier cluster.

    Args:
        data (np.array): Load shapes being clustered
        labels (list|np.array): A list defining what cluster each load shape belongs to
        score_choice (str): What scoring method to use
            Options: ['silhouette', 'silhouette_median', 'variance_ratio', 'calinski-harabasz', 'davies-bouldin']
        dist_metric (str): What distance metric to use when calculating distance
            Options: ['euclidean', 'seuclidean', 'manhattan', 'cosine']
        min_cluster_size (int): Minumum number of meters for a cluster
            Options: 2 < val

    Returns:
        score (float): Lower is better
        unable_to_calc_score (bool): Boolean that if true, means max score was used
    """

    # merge clusters to outlier cluster
    labels = merge_small_clusters(labels, min_cluster_size)

    # don't include outlier cluster in scoring
    idx = np.argwhere(labels != -1).flatten()

    non_outlier_cluster_count = labels.max() + 1
    if non_outlier_cluster_count < cluster_bound_lower:
        return get_max_score_from_system_size(), True

    if non_outlier_cluster_count > max_non_outlier_cluster_count:
        return get_max_score_from_system_size(), True

    if score_choice == _const.ScoreChoice.SILHOUETTE:
        try:
            # if sample size is a number then it randomly samples within the group, None looks at all
            # if using silhouette score for large datasets, might want to specify sample_size
            return (
                float(
                    sklearn.metrics.silhouette_score(
                        data[idx, :],
                        labels[idx],
                        metric=dist_metric,
                        sample_size=10_000,
                    )
                ),
                False,
            )

        except Exception:
            return float(10.0), True

    if score_choice == _const.ScoreChoice.SILHOUETTE_MEDIAN:
        try:
            # if this is too computationally intensive, could sample clusters instead
            score_all = -10 * sklearn.metrics.silhouette_samples(
                data, labels, metric=dist_metric
            )  # type: ignore
            return float(np.median(score_all[idx])), False

        except Exception:
            return float(10.0), True

    if score_choice in [
        _const.ScoreChoice.VARIANCE_RATIO,
        _const.ScoreChoice.CALINSKI_HARABASZ,
    ]:
        try:
            return (
                float(
                    -1
                    * sklearn.metrics.calinski_harabasz_score(data[idx, :], labels[idx])
                ),
                False,
            )
        except Exception:
            return get_max_score_from_system_size(), True

    if score_choice == _const.ScoreChoice.DAVIES_BOULDIN:
        try:
            return (
                float(sklearn.metrics.davies_bouldin_score(data[idx, :], labels[idx])),
                False,
            )

        except Exception:
            return get_max_score_from_system_size(), True

    raise ValueError(
        f"{score_choice} is not a recognized scoring function in clustering"
    )
