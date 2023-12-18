"""
module is responsible for creating clusters

"""

from __future__ import annotations

import copy

import attrs
import numpy as np
import pandas as pd

from gridmeter._clustering import (
    transform as _transform,
    treatment_fit as _fit,
    bisect_k_means,
    settings as _settings,
    scoring as _scoring,
    bounds as _bounds,
    data as _data,
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
    final renumbering which also adds 1
    valid clusters will be 1 -> max_cluster_num
    """

    clusters = _scoring.merge_small_clusters(
        clusters=clusters, min_cluster_size=min_cluster_size
    )
    return clusters


@attrs.define
class _LabelResult:
    """
    contains metrics about a cluster label returned from skitlearn
    """

    labels: np.ndarray
    score: float
    score_unable_to_be_calculated: bool
    n_clusters: int


def _get_all_label_results(
    data: np.ndarray,
    n_cluster_upper: int,
    cluster_bound_lower: int,
    score_choice: str,
    dist_metric: str,
    min_cluster_size: int,
    seed: int,
    max_non_outlier_cluster_count: int,
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
        data=data, n_clusters=n_cluster_upper, seed=seed
    )

    results = []
    for n_cluster, labels in labels_dict.items():
        if n_cluster < cluster_bound_lower:
            continue

        score, score_unable_to_be_calculated = _scoring.score_clusters(
            data=data,
            labels=labels,
            score_choice=score_choice,
            dist_metric=dist_metric,
            min_cluster_size=min_cluster_size,
            cluster_bound_lower=cluster_bound_lower,
            max_non_outlier_cluster_count=max_non_outlier_cluster_count,
        )
        labels = _final_cluster_renumber(
            clusters=labels, min_cluster_size=min_cluster_size
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
        cls,
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


def get_cluster_result_generator_from_upper_bound_n_cluster(
    pool_loadshape_transform_result: _transform.InitialPoolLoadshapeTransform,
    upper_bound_n_cluster: int,
    cluster_bound_lower: int,
    score_choice: str,
    dist_metric: str,
    min_cluster_size: int,
    cluster_key: str,
    data: np.ndarray | None,
    seed: int,
    max_non_outlier_cluster_count: int,
) -> Iterable[ClusterResultIntermediate]:
    """
    similar to the calculate_cluster_result function but
    returns a list of cluster results using the upper bound of clusters to look for.

    If this function does not raise an Exception then the returned list is
    guaranteed to contain a single ClusterResult even if its values are meaningless.
    """

    if pool_loadshape_transform_result.err_msg is not None:
        yield ClusterResultIntermediate(
            n_clusters=upper_bound_n_cluster,
            cluster_key=cluster_key,
            cluster_df=pd.DataFrame(),
            labels=np.array([]),
            pool_loadshape_transform_result=pool_loadshape_transform_result,
            score=_scoring.get_max_score_from_system_size(),
            score_unable_to_be_calculated=True,
            seed=seed,
        )

    if data is None:
        data = pool_loadshape_transform_result.fpca_result

    label_results = _get_all_label_results(
        data=data.values,
        n_cluster_upper=upper_bound_n_cluster,
        score_choice=score_choice,
        dist_metric=dist_metric,
        min_cluster_size=min_cluster_size,
        seed=seed,
        cluster_bound_lower=cluster_bound_lower,
        max_non_outlier_cluster_count=max_non_outlier_cluster_count,
    )

    if len(label_results) == 0:
        yield ClusterResultIntermediate(
            n_clusters=upper_bound_n_cluster,
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
    pool_loadshape_transform_result: _transform.InitialPoolLoadshapeTransform,
    upper_bound_n_cluster: int,
    cluster_bound_lower: int,
    score_choice: str,
    dist_metric: str,
    min_cluster_size: int,
    cluster_key: str,
    data: np.ndarray | None,
    seed: int,
    n_iter_cluster: int,
    max_non_outlier_cluster_count: int,
) -> Iterable[tuple[ClusterResultIntermediate, ClusterScoreElement]]:
    """
    same as generator but increases seed for each n_iter_cluster to choose a different starting point
    for the clustering.

    Meant to increase the chance of finding a better scored cluster
    """

    for seed_inc in range(n_iter_cluster):
        incremented_seed = seed + seed_inc

        cluster_res_gen = get_cluster_result_generator_from_upper_bound_n_cluster(
            pool_loadshape_transform_result=pool_loadshape_transform_result,
            upper_bound_n_cluster=upper_bound_n_cluster,
            score_choice=score_choice,
            dist_metric=dist_metric,
            min_cluster_size=min_cluster_size,
            cluster_key=cluster_key,
            data=data,
            seed=incremented_seed,
            cluster_bound_lower=cluster_bound_lower,
            max_non_outlier_cluster_count=max_non_outlier_cluster_count,
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
    min_cluster_size: int,
    num_cluster_bound_upper: int,
    num_cluster_bound_lower: int,
    score_choice: str,
    dist_metric: str,
    cluster_key: str,
    seed: int,
    n_iter_cluster: int,
    max_non_outlier_cluster_count: int,
) -> tuple[ClusterResultIntermediate, list[ClusterScoreElement]]:
    """
    function which performs bisecting kmeans clustering on the pool loadshapes
    using the provided values.

    The upper_bound is calculated and then a single clustering attempt occurs using that number.
    All the labels up to that number are saved and then scored.
    Then the best scored labels are returned so that the values can be
    """
    data = pool_loadshape_transform_result.fpca_result
    cluster_bound_lower, cluster_bound_upper = _bounds.get_cluster_bounds(
        data=data,
        min_cluster_size=min_cluster_size,
        num_cluster_bound_upper=num_cluster_bound_upper,
        num_cluster_bound_lower=num_cluster_bound_lower,
    )

    cluster_result_gen = _get_all_cluster_result_generator(
        pool_loadshape_transform_result=pool_loadshape_transform_result,
        upper_bound_n_cluster=cluster_bound_upper,
        score_choice=score_choice,
        dist_metric=dist_metric,
        min_cluster_size=min_cluster_size,
        cluster_key=cluster_key,
        data=data,
        seed=seed,
        n_iter_cluster=n_iter_cluster,
        cluster_bound_lower=cluster_bound_lower,
        max_non_outlier_cluster_count=max_non_outlier_cluster_count,
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


def _transform_cluster_loadshape(df_ls_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    applies the transform to the cluster loadshapes
    """
    df_ls_cluster_transformed = df_ls_cluster.apply(_transform._normalize_loadshape, axis=1)

    return df_ls_cluster_transformed


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

    agg_type: str
    dist_metric: str

    @classmethod
    def from_cluster_result_and_agg_type(
        cls,
        cluster_result: ClusterResultIntermediate,
        score_elements: list[ClusterScoreElement],
        agg_type: str,
        dist_metric: str,
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
                agg_type=agg_type,
                dist_metric=dist_metric,
            )

        cluster_loadshape_df = _get_cluster_ls(
            df_cp_ls=cluster_result.pool_loadshape_transform_result.concatenated_loadshapes,
            cluster_df=cluster_result.cluster_df,
            agg_type=agg_type,
        )

        cluster_loadshape_transformed_df = _transform_cluster_loadshape(
            df_ls_cluster=cluster_loadshape_df  # type: ignore
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
            dist_metric=dist_metric,
            agg_type=agg_type,
        )

    @classmethod
    def from_comparison_pool_loadshapes_and_settings(
        cls, df_cp: pd.DataFrame, s: _settings.Settings
    ):
        """
        classmethod for creating a ClusterMatcher instance by providing the comparison pool loadshapes to use and a settings instance.

        Will do all necessary transformations and clustering/scoring needed in order to return the instance
        of the class that is capable of assigning weights to treatment loadshapes.
        """
        ls_transform = _transform.InitialPoolLoadshapeTransform.from_full_cp_ls_df(
            df=df_cp, min_var_ratio=s.FPCA_MIN_VARIANCE_RATIO
        )

        best_scored_cluster, score_elements = get_best_scored_cluster_result(
            pool_loadshape_transform_result=ls_transform,
            cluster_key="",
            dist_metric=s.DIST_METRIC,
            max_non_outlier_cluster_count=s.MAX_NON_OUTLIER_CLUSTER_COUNT,
            min_cluster_size=s.MIN_CLUSTER_SIZE,
            n_iter_cluster=1,
            num_cluster_bound_lower=s.NUM_CLUSTER_BOUND_LOWER,
            num_cluster_bound_upper=s.NUM_CLUSTER_BOUND_UPPER,
            score_choice=s.SCORE_CHOICE,
            seed=s.SEED,
        )

        return ClusterResult.from_cluster_result_and_agg_type(
            cluster_result=best_scored_cluster,
            score_elements=score_elements,
            agg_type=s.AGG_TYPE,
            dist_metric=s.DIST_METRIC,
        )

    @property
    def cluster_loadshape_transformed_srs(self):
        """
        returns the series version of the dataframe to be used for matching
        """
        return self.cluster_loadshape_transformed_df["ls"]

    def get_match_treatment_to_cluster_df(
        self,
        treatment_loadshape_df: pd.DataFrame,
    ):
        """
        performs the matching logic to a provided treatment_loadshape dataframe

        TODO: Handle call when no valid scores were found?

        """

        transformed_treatment_loadshape = _fit._transform_treatment_loadshape(
            df=treatment_loadshape_df
        )

        df_ls_cluster = self.cluster_loadshape_transformed_df

        return _fit._match_treatment_to_cluster(
            df_ls_t=transformed_treatment_loadshape,
            df_ls_cluster=df_ls_cluster,
            agg_type=self.agg_type,
            dist_metric=self.dist_metric,
        )