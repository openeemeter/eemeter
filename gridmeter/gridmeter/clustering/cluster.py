"""
module is responsible for creating clusters

"""

from __future__ import annotations
from typing import Optional

import attrs
import numpy as np
import pandas as pd

from gridmeter.clustering import (
    transform as _transform,
    treatment_fit as _fit,
    bisect_k_means,
    settings as _settings,
    scoring as _scoring,
    bounds as _bounds,
)

from typing import Iterable

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_bisecting_kmeans_cluster_label_dict(
    data: np.ndarray, n_clusters: int, seed: int
) -> dict[int, np.ndarray]:
    """calls overridden class but returns labels_full which is a dictionary of
    n_clusters -> labels

    Called so only one clustering needs to occur and the scores can occur after
    """
    algo = bisect_k_means.BisectingKMeans(
        n_clusters=n_clusters,
        init="k-means++",  # does not benefit from k-means++ like other k-means
        n_init=3,  # default is 1
        random_state=seed,  # can be set to None or seed_num
        algorithm="elkan",  # ['lloyd', 'elkan']
        bisecting_strategy="largest_cluster",  # ['biggest_inertia', 'largest_cluster']
    )
    algo.fit(data)

    return algo.labels_full


def _final_cluster_renumber(clusters: np.ndarray, min_cluster_size: int):
    """
    final renumbering
    valid clusters will be 0 -> max_cluster_num
    """

    clusters = _scoring.merge_small_clusters(
        clusters=clusters, min_cluster_size=min_cluster_size
    )
    return clusters


@attrs.define
class _LabelResult:
    """
    contains metrics about a cluster label returned from sklearn
    """

    labels: np.ndarray
    score: float
    score_unable_to_be_calculated: bool
    n_clusters: int


def _get_all_label_results(
    data: np.ndarray,
    n_cluster_upper: int,
    n_cluster_lower: int,
    seed: int,
    s: _settings.Settings,
) -> list[_LabelResult]:
        
    if n_cluster_upper < 2:
        """
        occurs with following:
            len(data) -> 11, min_cluster_size -> 15, num_cluster_bound_upper -> 1,500
                calculated upper bound equals -1
        """
        return []

    if n_cluster_upper > len(data):
        return []

    if len(data) == 0:
        return []

    labels_dict = _get_bisecting_kmeans_cluster_label_dict(
        data=data, 
        n_clusters=n_cluster_upper, 
        seed=seed
    )

    results = []
    for n_cluster, labels in labels_dict.items():
        if n_cluster < n_cluster_lower:
            continue

        score, score_unable_to_be_calculated = _scoring.score_clusters(
            data=data,
            labels=labels,
            n_cluster_lower=n_cluster_lower,
            s=s,
        )
        labels = _final_cluster_renumber(
            clusters=labels, 
            min_cluster_size=s.MIN_CLUSTER_SIZE
        )
        label_res = _LabelResult(
            labels=labels,
            score=score,
            score_unable_to_be_calculated=score_unable_to_be_calculated,
            n_clusters=n_cluster,
        )
        results.append(label_res)

    return results


@attrs.define
class ClusterResultIntermediate:

    """
    dataclass which contains the information about the result
    of a clustering

    includes the number of clusters, the labels and the score
    """

    n_clusters: int
    score: float
    labels: np.ndarray
    cluster_df: pd.DataFrame
    score_unable_to_be_calculated: bool
    pool_loadshape_transform_result: _transform.InitialPoolLoadshapeTransform
    cluster_key: str
    seed: int

    @classmethod
    def from_label_result_and_pool_loadshape_transform_result(
        self,
        label_result: _LabelResult,
        pool_loadshape_transform_result: _transform.InitialPoolLoadshapeTransform,
        cluster_key: str,
        seed: int,
    ):
        """
        meant to be called on the list of label results from a single cluster of override class
        """
        if pool_loadshape_transform_result.err_msg is not None:
            raise ValueError(
                "programmer error. this check should be done before calculating all label results"
            )

        cluster_df = pd.DataFrame()
        if not label_result.score_unable_to_be_calculated:
            cluster_df = _create_cluster_dataframe(
                df_cp=pool_loadshape_transform_result.fpca_result,
                clusters=label_result.labels,
            )

        return ClusterResultIntermediate(
            n_clusters=label_result.n_clusters,
            labels=label_result.labels,
            cluster_df=cluster_df,
            cluster_key=cluster_key,
            score=label_result.score,
            pool_loadshape_transform_result=pool_loadshape_transform_result,
            score_unable_to_be_calculated=label_result.score_unable_to_be_calculated,
            seed=seed,
        )


def _create_cluster_dataframe(df_cp: pd.DataFrame, clusters: np.ndarray):
    cp_id = df_cp.index.get_level_values("id").unique().values
    cluster_df = pd.DataFrame(
        {"id": cp_id, "cluster": clusters}, columns=["id", "cluster"]
    )
    cluster_df = cluster_df.set_index("id").sort_values(by=["cluster"])
    return cluster_df


def get_all_cluster_results_generator(
    data: Optional[np.ndarray],
    pool_loadshape_transform_result: _transform.InitialPoolLoadshapeTransform,
    cluster_bound_upper: int,
    cluster_bound_lower: int,
    cluster_key: str,
    seed: int,
    s: _settings.Settings,
) -> Iterable[ClusterResultIntermediate]:
    """
    similar to the calculate_cluster_result function but
    returns a list of cluster results using the upper bound of clusters to look for.

    If this function does not raise an Exception then the returned list is
    guaranteed to contain a single ClusterResult even if its values are meaningless.
    """

    if pool_loadshape_transform_result.err_msg is not None:
        yield ClusterResultIntermediate(
            n_clusters=cluster_bound_upper,
            cluster_key=cluster_key,
            cluster_df=pd.DataFrame(),
            labels=np.array([]),
            pool_loadshape_transform_result=pool_loadshape_transform_result,
            score=_scoring.get_max_score_from_system_size(),
            score_unable_to_be_calculated=True,
            seed=seed,
        )

    # TODO: Why is this needed?
    if data is None:
        data = pool_loadshape_transform_result.fpca_result

    label_results = _get_all_label_results(
        data=data.values,
        n_cluster_upper=cluster_bound_upper,
        n_cluster_lower=cluster_bound_lower,
        seed=seed,
        s=s,
    )

    if len(label_results) == 0:
        yield ClusterResultIntermediate(
            n_clusters=cluster_bound_upper,
            cluster_key=cluster_key,
            cluster_df=pd.DataFrame(),
            labels=np.array([]),
            pool_loadshape_transform_result=pool_loadshape_transform_result,
            score=_scoring.get_max_score_from_system_size(),
            score_unable_to_be_calculated=True,
            seed=seed,
        )

    for label_result in label_results:
        cluster_result = ClusterResultIntermediate.from_label_result_and_pool_loadshape_transform_result(
            label_result=label_result,
            pool_loadshape_transform_result=pool_loadshape_transform_result,
            cluster_key=cluster_key,
            seed=seed,
        )
        yield cluster_result


def _iterate_best_found_cluster(cluster_results: Iterable[ClusterResultIntermediate]):
    """
    given an iterable of cluster_results, return the best scored.

    Fails if best_found is None as it should always be provided a single
    result at minimum even if the values are meaningless

    Meant to be used to find best score of a single starting seed
    and then for
    """
    best_scored_cluster = None

    for cluster_result in cluster_results:
        if best_scored_cluster is None:
            best_scored_cluster = cluster_result
            continue

        if (
            best_scored_cluster.score_unable_to_be_calculated
            and not cluster_result.score_unable_to_be_calculated
        ):
            best_scored_cluster = cluster_result
            continue

        if cluster_result.score_unable_to_be_calculated:
            continue

        if best_scored_cluster.score > cluster_result.score:
            best_scored_cluster = cluster_result
            continue

    if best_scored_cluster is None:
        raise ValueError("best scored cluster is None")

    return best_scored_cluster


@attrs.define
class ClusterScoreElement:
    """
    contains information about the best score of a group of cluster results.

    one score element per group

    should be calculated for each iteration of clustering (changing of seed)

    final cluster result should contain score elements for all iterations.

    This is so that n_iter_cluster can be set very high and all the information about
    what scores would have been in between can be captured. These elements are intended to be analyzed to
    determine the most reasonable value to use.

    This is important to determine because n_iter_cluster is a resource constraint and
    using the lowest satisfactory value is a direct performance increase.
    """

    iteration: int
    seed: int
    n_clusters: int
    score: float
    score_unable_to_be_calculated: bool


def _get_all_cluster_result_generator(
    data: Optional[np.ndarray],
    pool_loadshape_transform_result: _transform.InitialPoolLoadshapeTransform,
    cluster_bound_upper: int,
    cluster_bound_lower: int,
    cluster_key: str,
    n_iter_cluster: int,
    s: _settings.Settings,

) -> Iterable[tuple[ClusterResultIntermediate, ClusterScoreElement]]:
    """
    same as generator but increases seed for each n_iter_cluster to choose a different starting point
    for the clustering.

    Meant to increase the chance of finding a better scored cluster
    """

    for seed_inc in range(n_iter_cluster):
        incremented_seed = s.SEED + seed_inc

        cluster_res_gen = get_all_cluster_results_generator(
            data=data,
            pool_loadshape_transform_result=pool_loadshape_transform_result,
            cluster_bound_upper=cluster_bound_upper,
            cluster_bound_lower=cluster_bound_lower,
            cluster_key=cluster_key,
            seed=incremented_seed,
            s=s,
        )

        best_scored_cluster_for_seed = _iterate_best_found_cluster(
            cluster_results=cluster_res_gen
        )

        cluster_score_element = ClusterScoreElement(
            iteration=seed_inc + 1,
            n_clusters=best_scored_cluster_for_seed.n_clusters,
            score=best_scored_cluster_for_seed.score,
            score_unable_to_be_calculated=best_scored_cluster_for_seed.score_unable_to_be_calculated,
            seed=incremented_seed,
        )

        yield best_scored_cluster_for_seed, cluster_score_element


def get_best_scored_cluster_result(
    pool_loadshape_transform_result: _transform.InitialPoolLoadshapeTransform,
    cluster_key: str,
    n_iter_cluster: int,
    s: _settings.Settings,
) -> tuple[ClusterResultIntermediate, list[ClusterScoreElement]]:
    """
    function which performs bisecting kmeans clustering on the pool loadshapes
    using the provided values.

    The upper_bound is calculated and then a single clustering attempt occurs using that number.
    All the labels up to that number are saved and then scored.
    Then the best scored labels are returned so that the values can be
    """

    fpca_data = pool_loadshape_transform_result.fpca_result

    data_size = np.shape(fpca_data)[0]
    cluster_bound_lower, cluster_bound_upper = _bounds.get_cluster_bounds(
        data_size=data_size,
        min_cluster_size=s.MIN_CLUSTER_SIZE,
        num_cluster_bound_upper=s.NUM_CLUSTER_BOUND_UPPER,
        num_cluster_bound_lower=s.NUM_CLUSTER_BOUND_LOWER,
    )

    cluster_result_gen = _get_all_cluster_result_generator(
        data=fpca_data,
        pool_loadshape_transform_result=pool_loadshape_transform_result,
        cluster_bound_upper=cluster_bound_upper,
        cluster_bound_lower=cluster_bound_lower,
        cluster_key=cluster_key,
        n_iter_cluster=n_iter_cluster,
        s=s,
    )

    best_scored_cluster = None
    score_elements: list[ClusterScoreElement] = []
    for cluster_result_tup in cluster_result_gen:
        cluster_result, score_element = cluster_result_tup
        score_elements.append(score_element)

        if best_scored_cluster is None:
            best_scored_cluster = cluster_result
            continue

        best_scored_cluster = _iterate_best_found_cluster(
            [best_scored_cluster, cluster_result]
        )

    if best_scored_cluster is None:
        raise Exception(
            "best_scored_cluster is None. This should only ever occur if somehow no clustering was performed. Likely settings/logic error."
        )

    return best_scored_cluster, score_elements


def _get_cluster_ls(df_cp_ls: pd.DataFrame, cluster_df: pd.DataFrame, agg_type: str):
    """
    original cp loadshape and cluster df
    settings for agg_type
    """

    cluster_df = cluster_df.join(df_cp_ls, on="id")
    cluster_df = cluster_df.reset_index().set_index(["id", "cluster"])  # type: ignore

    # calculate cp_df
    df_cluster_ls = cluster_df.groupby("cluster").agg(agg_type)  # type: ignore
    cluster_ls = df_cluster_ls[df_cluster_ls.index.get_level_values(0) > -1]  # don't match to outlier cluster

    return cluster_ls


@attrs.define
class ClusterResult:
    """
    ClusterResult is the final result of providing any configurable settings
    values and a set of loadshapes used as a comparison pool to cluster.

    Contains metrics about the result such as the calculated score and the best scores of each clustering iteration
    that used a different seed/starting point.

    Additionally contains dataframes which are required to perform the matching logic on any loadshape that is to be weighted against the result.
    """

    cluster_key: str
    cluster_loadshape_transformed_df: pd.DataFrame

    cluster_df: pd.DataFrame

    n_clusters: int
    score: float
    score_unable_to_be_calculated: bool
    seed: int

    iter_scores: tuple[ClusterScoreElement, ...]

    s: _settings.Settings

    @classmethod
    def from_cluster_result_and_agg_type(
        cls,
        cluster_result: ClusterResultIntermediate,
        score_elements: list[ClusterScoreElement],
        s: _settings.Settings,
    ):
        """
        classmethod to create the final cluster result which can be used
        to match treatment models/apply weights

        This is meant to be called once the best scored cluster_result is determined.
        This allows the calculation of the cluster loadshape to happen only once
        """
        if cluster_result.score_unable_to_be_calculated:
            return ClusterResult(
                cluster_key=cluster_result.cluster_key,
                cluster_df=pd.DataFrame(),
                cluster_loadshape_transformed_df=pd.DataFrame(),
                n_clusters=cluster_result.n_clusters,
                score=cluster_result.score,
                score_unable_to_be_calculated=True,
                seed=cluster_result.seed,
                iter_scores=tuple(score_elements),
                s=s,
            )

        cluster_loadshape_df = _get_cluster_ls(
            df_cp_ls=cluster_result.pool_loadshape_transform_result.concatenated_loadshapes,
            cluster_df=cluster_result.cluster_df,
            agg_type=s.AGG_TYPE,
        )

        cluster_loadshape_transformed_df = _transform._normalize_df_loadshapes(
            df=cluster_loadshape_df,
            s=s
        )

        return ClusterResult(
            # cluster_result=cluster_result,
            cluster_loadshape_transformed_df=cluster_loadshape_transformed_df,
            cluster_df=cluster_result.cluster_df,
            n_clusters=cluster_result.n_clusters,
            score=cluster_result.score,
            score_unable_to_be_calculated=cluster_result.score_unable_to_be_calculated,
            cluster_key=cluster_result.cluster_key,
            seed=cluster_result.seed,
            iter_scores=tuple(score_elements),
            s=s,
        )

    @classmethod
    def from_comparison_pool_loadshapes_and_settings(
        cls, 
        df_cp: pd.DataFrame, 
        s: _settings.Settings
    ):
        """
        classmethod for creating a ClusterMatcher instance by providing the comparison pool loadshapes to use and a settings instance.

        Will do all necessary transformations and clustering/scoring needed in order to return the instance
        of the class that is capable of assigning weights to treatment loadshapes.
        """
        ls_transform = _transform.InitialPoolLoadshapeTransform(
            df=df_cp,
            s=s,
        )

        best_scored_cluster, score_elements = get_best_scored_cluster_result(
            pool_loadshape_transform_result=ls_transform, 
            cluster_key="",
            n_iter_cluster=1,
            s=s,
        )

        return ClusterResult.from_cluster_result_and_agg_type(
            cluster_result=best_scored_cluster,
            score_elements=score_elements,
            s=s,
        )

    def get_match_treatment_to_cluster_df(
        self,
        treatment_loadshape_df: pd.DataFrame,
        s: _settings.Settings,
    ):
        """
        performs the matching logic to a provided treatment_loadshape dataframe

        TODO: Handle call when no valid scores were found?

        """

        transformed_treatment_loadshape = _transform._normalize_df_loadshapes(
            df=treatment_loadshape_df,
            s=s
        )

        df_t_coeffs = _fit._match_treatment_to_cluster(
            df_ls_t=transformed_treatment_loadshape,
            df_ls_cluster=self.cluster_loadshape_transformed_df,
            s=s,
        )

        return df_t_coeffs