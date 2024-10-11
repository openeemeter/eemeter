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

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

from tslearn.clustering import TimeSeriesKMeans, silhouette_score

from eemeter.eemeter.models.hourly.model import HourlyModel
from eemeter.eemeter.models.hourly import settings as _settings


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
            HoF = {"score": -np.inf, "clusters": None}
            for n_cluster in range(2, settings.MAX_CLUSTER_COUNT + 1):
                km = TimeSeriesKMeans(
                    n_clusters          = n_cluster,
                    max_iter            = settings.MAX_ITER,
                    tol                 = settings.TOL,
                    n_init              = settings.N_INIT,
                    metric              = settings.METRIC,
                    max_iter_barycenter = settings.MAX_ITER_BARYCENTER,
                    init                = settings.INIT_METHOD,
                    random_state        = settings._SEED,
                )
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels,
                    metric = settings.METRIC,
                    sample_size = settings.SCORE_SAMPLE_SIZE,
                    random_state = settings._SEED,
                )

                if score > HoF["score"]:
                    HoF["score"] = score
                    # HoF["n_cluster"] = n_cluster
                    # HoF["km"] = km
                    HoF["clusters"] = labels
                    # HoF["centroids"] = km.cluster_centers_

            df_temporal_clusters = pd.DataFrame(
                HoF["clusters"].astype(int),
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