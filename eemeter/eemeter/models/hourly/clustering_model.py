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

import numpy as np
import pandas as pd

from tslearn.clustering import TimeSeriesKMeans, silhouette_score

from eemeter.eemeter.models.hourly.model import HourlyModel


class ClusteringHourlyModel(HourlyModel):
    def _add_temperature_bins(self, df):
        # add mean daily temperature catagory
        if self.settings.INCLUDE_TEMPERATURE_BINS: #TODO: we need to take care of empty bins (in prediction?), if there is any.
            daily_temp = df.groupby('date')["temperature"].mean()
            df = pd.merge(df, daily_temp, on='date', how='left').rename(columns={'temperature_x': 'temperature', 'temperature_y': 'daily_temp'}).set_index(df.index)
            #same size bin cut
            if self.fit_pred_status == 'fitting':
                if not self.same_size_bin:
                    res, temp_bins= pd.qcut(df['daily_temp'], q=self.n_bins, retbins=True, labels=False)
                else:
                    res, temp_bins= pd.cut(df['daily_temp'], bins=self.n_bins, retbins=True, labels=False)

                temp_bins = list(temp_bins)
                temp_bins[0] = -np.inf
                temp_bins[-1] = np.inf
                # temp_bins.insert(0, -np.inf)
                # temp_bins.append(np.inf)
                res, temp_bins= pd.cut(df['daily_temp'], bins=temp_bins, retbins=True, labels=False)
                df['daily_temp_bins_cat'] = res
                self.daily_temp_bins = temp_bins

            elif self.fit_pred_status == 'predicting':
                res = pd.cut(df['daily_temp'], bins=self.daily_temp_bins, labels=False)
                df['daily_temp_bins_cat'] = res


            bin_dummies = pd.get_dummies(
                pd.Categorical(df['daily_temp_bins_cat'], categories=range(len(self.daily_temp_bins )-1)),
                prefix='daily_temp'
                )
            bin_dummies.index = df.index

            temp_cat = [f'daily_temp_{i}' for i in range(len(self.daily_temp_bins)-1)]
            self.categorical_features.extend(temp_cat)
            df = pd.merge(df, bin_dummies, how='left', left_index=True, right_index=True)

        return df


    def _add_categorical_features(self, df):
        self.categorical_features = []

        df["date"] = df.index.date
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek
        df["hour"] = df.index.hour

        if self.fit_pred_status == 'fitting':
            fit_df_grouped = df.groupby(['month', 'day_of_week', 'hour'])['observed'].mean().reset_index()
            fit_grouped = fit_df_grouped.groupby(['month', 'day_of_week'])['observed'].apply(list)

            X = np.array(fit_grouped.tolist())

            max_clusters = 6 # TODO: add to settings with valid entries
            metric = 'euclidean' # TODO: add to settings with valid entries
            seed = self.settings._SEED

            HoF = {"score": -np.inf, "n_cluster": 0, "km": None, "clusters": None, "centroids": None}
            for n_cluster in range(2, max_clusters + 1):
                km = TimeSeriesKMeans(
                    n_clusters=n_cluster, 
                    metric=metric, 
                    random_state=seed,
                )
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels)

                if score > HoF["score"]:
                    HoF["score"] = score
                    HoF["n_cluster"] = n_cluster
                    HoF["km"] = km
                    HoF["clusters"] = labels
                    HoF["centroids"] = km.cluster_centers_

            self.n_clusters = HoF["n_cluster"]
            self.km = HoF["km"]
            self.clusters = HoF["clusters"]
            self.centroids = HoF["centroids"]

            self.fit_df_grouped_index = fit_grouped.index
            df['cluster'] = df.apply(lambda x: self.clusters[self.fit_df_grouped_index.get_loc((x['month'], x['day_of_week']))], axis=1) #TODO: important: what if we didn't have one month in the baseline

        elif self.fit_pred_status == 'predicting':
            pred_df_grouped = df.groupby(['month', 'day_of_week', 'hour'])['observed'].mean().reset_index()
            pred_grouped = pred_df_grouped.groupby(['month', 'day_of_week'])['observed'].apply(list)
            # if number of day/month is the same between baseline/prediction, we can use the same clusters
            # TODO: Remove try/except and handle the case where the number of day/month is different
            try:
                df['cluster'] = df.apply(lambda x: self.clusters[pred_grouped.index.get_loc((x['month'], x['day_of_week']))], axis=1) #TODO: important: what if we didn't have one month in the baseline
            except:
                unique_pairs = self.fit_df_grouped_index.unique()
                unique_pairs = [f"{month}-{day}" for month, day in unique_pairs]
                df['month_day_pair'] = df['month'].astype(str) + '-' + df['day_of_week'].astype(str)
                filtered_df = df[df['month_day_pair'].isin(unique_pairs)]
                filtered_df = filtered_df.drop(columns=['month_day_pair'])
                df['cluster'] = None

                df.loc[filtered_df.index, 'cluster'] = filtered_df.apply(lambda x: self.clusters[self.fit_df_grouped_index.get_loc((x['month'], x['day_of_week']))], axis=1)
                # for the rest of the df, for each day compare the observed with the cluster centroids and assign the cluster with the smallest distance
                not_seen = df[df['cluster'].isna()]
                not_seen_grouped = not_seen.groupby(['month', 'day_of_week', 'hour'])['observed'].mean().reset_index()
                not_seen_grouped = not_seen_grouped.groupby(['month', 'day_of_week'])['observed'].apply(list)
                X = np.array(not_seen_grouped.tolist())
                not_seen_clusters = self.km.predict(X)
                not_seen['cluster'] = not_seen.apply(lambda x: not_seen_clusters[not_seen_grouped.index.get_loc((x['month'], x['day_of_week']))], axis=1)
                df.loc[not_seen.index, 'cluster'] = not_seen['cluster']

        cluster_dummies = pd.get_dummies(
            pd.Categorical(df['cluster'], categories=range(self.n_clusters)),
            prefix='cluster'
        )
        cluster_dummies.index = df.index

        cluster_cat = [f"cluster_{i}" for i in range(self.n_clusters)]
        self.categorical_features.extend(cluster_cat)
        df = pd.merge(df, cluster_dummies, how='left', left_index=True, right_index=True)

        df = self._add_temperature_bins(df)

        return df