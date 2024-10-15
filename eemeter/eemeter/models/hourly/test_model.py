#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import sklearn
sklearn.set_config(assume_finite=True, skip_parameter_validation=True) # set to True, faster, we do checking, need to apply everywhere

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from tslearn.clustering import TimeSeriesKMeans, silhouette_score

from eemeter.eemeter.models.hourly.model import HourlyModel
from eemeter.eemeter.models.hourly import settings as _settings

from eemeter.common.clustering import (
    transform as _transform,
    bisect_k_means as _bisect_k_means,
    scoring as _scoring,
)
from pydantic import BaseModel


class TestHourlyModel(HourlyModel):
    def _add_categorical_features(self, df):
        def set_initial_temporal_clusters(df):
            fit_df_grouped = (
                df.groupby(["month", "day_of_week", "hour_of_day"])["observed"]
                .mean()
                .reset_index()
            )
            fit_grouped = fit_df_grouped.groupby(["month", "day_of_week"])[
                "observed"
            ].apply(np.array)

            # convert fit_grouped to 2D numpy array
            X = np.stack(fit_grouped.values, axis=0)

            settings = self.settings.TEMPORAL_CLUSTER
            labels = cluster_temporal_features(
                pd.DataFrame(X), 
                settings.FPCA_MIN_VARIANCE_RATIO_EXPLAINED, 
                settings.RECLUSTER_COUNT, 
                settings.N_CLUSTER_LOWER, 
                settings.N_CLUSTER_UPPER, 
                settings.SCORE_METRIC, 
                settings.DISTANCE_METRIC, 
                settings.MIN_CLUSTER_SIZE, 
                200, 
                settings._SEED
            )

            df_temporal_clusters = pd.DataFrame(
                labels,
                columns=["temporal_cluster"],
                index=fit_grouped.index,
            )

            return df_temporal_clusters

        def correct_missing_temporal_clusters(df):
            # check and match any missing temporal combinations

            # get all unique combinations of month and day_of_week in df
            df_temporal = df[["month", "day_of_week"]].drop_duplicates()
            df_temporal = df_temporal.sort_values(["month", "day_of_week"])
            df_temporal_index = df_temporal.set_index(["month", "day_of_week"]).index

            # reindex self.df_temporal_clusters to df_temporal_index
            df_temporal_clusters = self._df_temporal_clusters.reindex(df_temporal_index)

            # get index of any nan values in df_temporal_clusters
            missing_combinations = df_temporal_clusters[
                df_temporal_clusters["temporal_cluster"].isna()
            ].index
            if not missing_combinations.empty:
                # TODO: this assumes that observed has values in df and not all null
                if "observed" in df.columns:
                    # filter df to only include missing combinations
                    df_missing = df[
                        df.set_index(["month", "day_of_week"]).index.isin(
                            missing_combinations
                        )
                    ]

                    df_missing_grouped = (
                        df_missing.groupby(["month", "day_of_week", "hour_of_day"])["observed"]
                        .mean()
                        .reset_index()
                    )
                    df_missing_grouped = df_missing_grouped.groupby(
                        ["month", "day_of_week"]
                    )["observed"].apply(np.array)

                    # convert fit_grouped to 2D numpy array
                    X = np.stack(df_missing_grouped.values, axis=0)

                    # calculate average observed for known clusters
                    # join df_temporal_clusters to df on month and day_of_week
                    df = pd.merge(
                        df,
                        df_temporal_clusters,
                        how="left",
                        left_on=["month", "day_of_week"],
                        right_index=True,
                    )

                    df_known = df[
                        ~df.set_index(["month", "day_of_week"]).index.isin(
                            missing_combinations
                        )
                    ]

                    df_known_groupby = df_known.groupby(
                        ["month", "day_of_week", "hour_of_day"]
                    )["observed"]
                    df_known_mean = df_known_groupby.mean().reset_index()
                    df_known_mean = df_known_mean.groupby(["month", "day_of_week"])[
                        "observed"
                    ].apply(np.array)

                    # get temporal clusters df_known
                    temporal_clusters = df_known.groupby(["month", "day_of_week"])[
                        "temporal_cluster"
                    ].first()
                    temporal_clusters = temporal_clusters.reindex(df_known_mean.index)

                    X_known = np.stack(df_known_mean.values, axis=0)

                    # get smallest distance between X and X_known
                    dist = cdist(X, X_known, metric="euclidean")
                    min_dist_idx = np.argmin(dist, axis=1)

                    # set labels to minimum distance of known clusters
                    labels = temporal_clusters.iloc[min_dist_idx].values
                    df_temporal_clusters.loc[
                        missing_combinations, "temporal_cluster"
                    ] = labels

                    self._df_temporal_clusters = df_temporal_clusters

                else:
                    # TODO: There's better ways of handling this
                    # unstack and fill missing days in each month
                    # assuming months more important than days
                    df_temporal_clusters = df_temporal_clusters.unstack()

                    # fill missing days in each month
                    df_temporal_clusters = df_temporal_clusters.ffill(axis=1)
                    df_temporal_clusters = df_temporal_clusters.bfill(axis=1)

                    # fill missing months if any remaining empty
                    df_temporal_clusters = df_temporal_clusters.ffill(axis=0)
                    df_temporal_clusters = df_temporal_clusters.bfill(axis=0)

                    df_temporal_clusters = df_temporal_clusters.stack()

            return df_temporal_clusters

        # assign basic temporal features
        df["date"] = df.index.date
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek
        df["hour_of_day"] = df.index.hour

        # assign temporal clusters
        if not self.is_fit:
            self._df_temporal_clusters = set_initial_temporal_clusters(df)
        else:
            self._df_temporal_clusters = correct_missing_temporal_clusters(df)

        # join df_temporal_clusters to df
        df = pd.merge(
            df,
            self._df_temporal_clusters,
            how="left",
            left_on=["month", "day_of_week"],
            right_index=True,
        )
        n_clusters = self._df_temporal_clusters["temporal_cluster"].nunique()

        cluster_dummies = pd.get_dummies(
            pd.Categorical(df["temporal_cluster"], categories=range(n_clusters)),
            prefix="temporal_cluster",
        )
        cluster_dummies.index = df.index

        cluster_cat = [f"temporal_cluster_{i}" for i in range(n_clusters)]
        self._categorical_features = cluster_cat

        df = pd.merge(
            df, cluster_dummies, how="left", left_index=True, right_index=True
        )

        if self.settings.TEMPERATURE_BIN is not None:
            df, temp_bin_cols = self._add_temperature_bins(df)
            self._categorical_features.extend(temp_bin_cols)

        return df
    

class _LabelResult(BaseModel):
    """
    contains metrics about a cluster label returned from sklearn
    """
    class Config:
        arbitrary_types_allowed = True

    labels: np.ndarray
    score: float
    score_unable_to_be_calculated: bool
    n_clusters: int


def cluster_temporal_features(
    df: pd.DataFrame,
    min_var_ratio: float,
    recluster_count: int,
    n_cluster_lower: int,
    n_cluster_upper: int,
    score_choice: str,
    dist_metric: str,
    min_cluster_size: int,
    max_non_outlier_cluster_count: int,
    seed: int,
):
    """
    clusters the temporal features of the dataframe
    """
    # get the functional principal component analysis of the load
    df_fpca, err = _transform._get_fpca_from_loadshape(df, min_var_ratio)

    results = []
    for i in range(recluster_count):
        algo = _bisect_k_means.BisectingKMeans(
            n_clusters=n_cluster_upper,
            init="k-means++",  # does not benefit from k-means++ like other k-means
            n_init=5,  # default is 1
            random_state=seed + i,  # can be set to None or seed_num
            algorithm="elkan",  # ['lloyd', 'elkan']
            bisecting_strategy="largest_cluster",  # ['biggest_inertia', 'largest_cluster']
        )
        algo.fit(df_fpca)
        labels_dict = algo.labels_full

        for n_cluster, labels in labels_dict.items():
            score, score_unable_to_be_calculated = _scoring.score_clusters(
                df_fpca.values,
                labels,
                n_cluster_lower,
                score_choice,
                dist_metric,
                min_cluster_size,
                max_non_outlier_cluster_count,
            )

            label_res = _LabelResult(
                labels=labels,
                score=score,
                score_unable_to_be_calculated=score_unable_to_be_calculated,
                n_clusters=n_cluster,
            )
            results.append(label_res)

    # get the results index with the smallest score
    HoF = None
    for result in results:
        if result.score_unable_to_be_calculated:
            continue

        if HoF is None or result.score < HoF.score:
            HoF = result

    return HoF.labels