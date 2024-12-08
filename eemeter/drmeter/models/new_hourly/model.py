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

from pydantic import BaseModel

import numpy as np
import pandas as pd

import sklearn
sklearn.set_config(assume_finite=True, skip_parameter_validation=True) # Faster, we do checking

from scipy.sparse import csr_matrix

from scipy.spatial.distance import cdist

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA

import pywt

from timeit import default_timer as timer

import json

from eemeter.eemeter.models.hourly.model import (
    HourlyModel,
    cluster_temporal_features,
    fit_exp_growth_decay,
)
from eemeter.drmeter.models.new_hourly import settings as _settings
from eemeter.drmeter.models.new_hourly import HourlyBaselineData, HourlyReportingData
from eemeter.common.clustering import (
    bisect_k_means as _bisect_k_means,
    scoring as _scoring,
)
from eemeter.common.metrics import BaselineMetrics, BaselineMetricsFromDict


# TODO: need to make explicit solar/nonsolar versions and set settings requirements/defaults appropriately
class DRHourlyModel(HourlyModel):
    """Note:
        Despite the temporal clusters, we can view all models created as a subset of the same full model.
        The temporal clusters would simply have the same coefficients within the same days combinations.
    """

    def __init__(
        self,
        settings: (
            _settings.HourlyNonSolarSettings | _settings.HourlySolarSettings | None
        ) = None,
    ):
        """ """

        # Initialize settings
        if settings is None:
            self.settings = _settings.HourlyNonSolarSettings()
        else:
            self.settings = settings

        # Initialize model
        if self.settings.SCALING_METHOD == _settings.ScalingChoice.STANDARDSCALER:
            self._feature_scaler = StandardScaler()
            self._y_scaler = StandardScaler()
        elif self.settings.SCALING_METHOD == _settings.ScalingChoice.ROBUSTSCALER:
            self._feature_scaler = RobustScaler(unit_variance=True)
            self._y_scaler = RobustScaler(unit_variance=True)

        self._T_edge_bin_coeffs = None

        self._model = ElasticNet(
            alpha=self.settings.ELASTICNET.ALPHA,
            l1_ratio=self.settings.ELASTICNET.L1_RATIO,
            fit_intercept=self.settings.ELASTICNET.FIT_INTERCEPT,
            precompute=self.settings.ELASTICNET.PRECOMPUTE,
            max_iter=self.settings.ELASTICNET.MAX_ITER,
            tol=self.settings.ELASTICNET.TOL,
            selection=self.settings.ELASTICNET.SELECTION,
            random_state=self.settings.ELASTICNET._SEED,
        )

        self._T_bin_edges = None
        self._T_edge_bin_rate = None
        self._df_temporal_clusters = None
        self._ts_features = self.settings._TRAIN_FEATURES.copy()
        self._categorical_features = None
        self._ts_feature_norm = None

        self.is_fit = False
        self.baseline_metrics = None

    def fit(self, baseline_data, ignore_disqualification=False):
        # if not isinstance(baseline_data, HourlyBaselineData):
        #     raise TypeError("baseline_data must be a DailyBaselineData object")
        # TODO check DQ, log warnings
        self._fit(baseline_data)
        
        return self
    
    def predict(
        self,
        reporting_data,
        ignore_disqualification=False,
    ):
        """Perform initial sufficiency and typechecks before passing to private predict"""
        if not self.is_fit:
            raise RuntimeError("Model must be fit before predictions can be made.")

        # TODO check DQ, log warnings

        # if not isinstance(reporting_data, (HourlyBaselineData, HourlyReportingData)):
        #     raise TypeError(
        #         "reporting_data must be a DailyBaselineData or DailyReportingData object"
        #     )

        df_eval = self._predict(reporting_data)

        return df_eval

    def _add_temperature_bins(self, df):
        # TODO: do we need to do something about empty bins in prediction? I think not but maybe
        settings = self.settings.TEMPERATURE_BIN

        # add temperature bins based on temperature
        if not self.is_fit:
            if settings.METHOD == "equal_sample_count":
                T_bin_edges = pd.qcut(df["temperature"], q=settings.N_BINS, labels=False)

            elif settings.METHOD == "equal_bin_width":
                T_bin_edges = pd.cut(df["temperature"], bins=settings.N_BINS, labels=False)

            elif settings.METHOD == "set_bin_width":
                bin_width = settings.BIN_WIDTH

                min_temp = np.floor(df["temperature"].min())
                max_temp = np.ceil(df["temperature"].max())

                if not settings.INCLUDE_EDGE_BINS:
                    step_num = np.round((max_temp - min_temp) / bin_width).astype(int) + 1

                    # T_bin_edges = np.arange(min_temp, max_temp + bin_width, bin_width)
                    T_bin_edges = np.linspace(min_temp, max_temp, step_num)

                else:
                    set_edge_bin_width = False
                    if set_edge_bin_width:
                        edge_bin_width = bin_width*1/2

                        bin_range = [min_temp + edge_bin_width, max_temp - edge_bin_width]

                    else:
                        edge_bin_count = int(len(df) * settings.EDGE_BIN_PERCENT)

                        # get 5th smallest and 5th largest temperatures
                        sorted_temp = np.sort(df["temperature"])
                        min_temp_reg_bin = np.ceil(sorted_temp[edge_bin_count])
                        max_temp_reg_bin = np.floor(sorted_temp[-edge_bin_count])

                        bin_range = [min_temp_reg_bin, max_temp_reg_bin]

                    step_num = np.round((bin_range[1] - bin_range[0]) / bin_width).astype(int) + 1

                    # create bins with set width
                    T_bin_edges = np.array([min_temp, *np.linspace(*bin_range, step_num), max_temp])

            else:
                raise ValueError("Invalid temperature binning method")

            # set the first and last bin to -inf and inf
            T_bin_edges[0] = -np.inf
            T_bin_edges[-1] = np.inf

            # store bin edges for prediction
            self._T_bin_edges = T_bin_edges

        T_bins = pd.cut(df["temperature"], bins=self._T_bin_edges, labels=False)

        df["temp_bin"] = T_bins

        # Create dummy variables for temperature bins
        bin_dummies = pd.get_dummies(
            pd.Categorical(
                df["temp_bin"], categories=range(len(self._T_bin_edges) - 1)
            ),
            prefix="temp_bin",
        )
        bin_dummies.index = df.index

        col_names = bin_dummies.columns.tolist()
        df = pd.merge(df, bin_dummies, how="left", left_index=True, right_index=True)

        return df, col_names

    def _add_categorical_features(self, df):
        def set_initial_temporal_clusters(df):
            fit_df_grouped = (
                df.groupby(["day_of_week", "hour_of_day"])["observed"]
                .mean()
                .reset_index()
            )
            fit_grouped = fit_df_grouped.groupby(["day_of_week"])[
                "observed"
            ].apply(np.array)

            # convert fit_grouped to 2D numpy array
            X = np.stack(fit_grouped.values, axis=0)

            settings = self.settings.TEMPORAL_CLUSTER
            labels = cluster_temporal_features(
                X,
                settings.WAVELET_N_LEVELS,
                settings.WAVELET_NAME,
                settings.WAVELET_MODE, 
                settings.PCA_MIN_VARIANCE_RATIO_EXPLAINED, 
                settings.RECLUSTER_COUNT, 
                settings.N_CLUSTER_LOWER, 
                settings.N_CLUSTER_UPPER, 
                settings.SCORE_METRIC, 
                settings.DISTANCE_METRIC, 
                settings.MIN_CLUSTER_SIZE, 
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
            df_temporal = df[["day_of_week"]].drop_duplicates()
            df_temporal = df_temporal.sort_values(["day_of_week"])
            df_temporal_index = df_temporal.set_index(["day_of_week"]).index

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
                        df.set_index(["day_of_week"]).index.isin(
                            missing_combinations
                        )
                    ]

                    df_missing_grouped = (
                        df_missing.groupby(["day_of_week", "hour_of_day"])["observed"]
                        .mean()
                        .reset_index()
                    )
                    df_missing_grouped = df_missing_grouped.groupby(
                        ["day_of_week"]
                    )["observed"].apply(np.array)

                    # convert fit_grouped to 2D numpy array
                    X = np.stack(df_missing_grouped.values, axis=0)

                    # calculate average observed for known clusters
                    # join df_temporal_clusters to df on day_of_week
                    df = pd.merge(
                        df,
                        df_temporal_clusters,
                        how="left",
                        left_on=["day_of_week"],
                        right_index=True,
                    )

                    df_known = df[
                        ~df.set_index(["day_of_week"]).index.isin(
                            missing_combinations
                        )
                    ]

                    df_known_groupby = df_known.groupby(
                        ["day_of_week", "hour_of_day"]
                    )["observed"]
                    df_known_mean = df_known_groupby.mean().reset_index()
                    df_known_mean = df_known_mean.groupby(["day_of_week"])[
                        "observed"
                    ].apply(np.array)

                    # get temporal clusters df_known
                    temporal_clusters = df_known.groupby(["day_of_week"])[
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
            left_on=["day_of_week"],
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

    def _add_supplemental_features(self, df):
        # TODO: should either do upper or lower on all strs
        if self.settings.SUPPLEMENTAL_TIME_SERIES_COLUMNS is not None:
            for col in self.settings.SUPPLEMENTAL_TIME_SERIES_COLUMNS:
                if (col in df.columns) and (col not in self._ts_features):
                    self._ts_features.append(col)

        if self.settings.SUPPLEMENTAL_CATEGORICAL_COLUMNS is not None:
            for col in self.settings.SUPPLEMENTAL_CATEGORICAL_COLUMNS:
                if (
                    (col in df.columns)
                    and (col not in self._ts_features)
                    and (col not in self._categorical_features)
                ):
                    self._categorical_features.append(col)

    def _sort_features(self, ts_features=None, cat_features=None):
        features = {"ts": ts_features, "cat": cat_features}

        # sort features
        for _type in ["ts", "cat"]:
            feat = features[_type]

            if feat is not None:
                sorted_cols = []
                for col in self._priority_cols[_type]:
                    cat_cols = [c for c in feat if c.startswith(col)]
                    sorted_cols.extend(sorted(cat_cols))

                # get all columns in self._categorical_feature not in sorted_cat_cols
                leftover_cols = [c for c in feat if c not in sorted_cols]
                if leftover_cols:
                    sorted_cols.extend(sorted(leftover_cols))

                features[_type] = sorted_cols

        return features["ts"], features["cat"]

    # TODO rename to avoid confusion with data sufficiency
    def _daily_sufficiency(self, df):
        # remove days with insufficient data
        min_hours = self.settings.MIN_DAILY_TRAINING_HOURS

        if min_hours > 0:
            # find any rows with interpolated data
            cols = [col for col in df.columns if col.startswith("interpolated_")]
            df["interpolated"] = df[cols].any(axis=1)

            # if row contains any null values, set interpolated to True
            df["interpolated"] = df["interpolated"] | df.isnull().any(axis=1)

            # count number of non interpolated hours per day
            daily_hours = 24 - df.groupby("date")["interpolated"].sum()
            sufficient_days = daily_hours[daily_hours >= min_hours].index

            # set "include_day" column to True if day has sufficient hours
            df["include_date"] = df["date"].isin(sufficient_days)

        else:
            df["include_date"] = True

        return df

    def _normalize_features(self, df):
        """ """
        train_features = self._ts_features
        self._ts_feature_norm = [i + "_norm" for i in train_features]

        # need to set scaler if not fit
        if not self.is_fit:
            self._feature_scaler.fit(df[train_features])
            self._y_scaler.fit(df["observed"].values.reshape(-1, 1))

        data_transformed = self._feature_scaler.transform(df[train_features])
        normalized_df = pd.DataFrame(
            data_transformed, index=df.index, columns=self._ts_feature_norm
        )

        df = pd.concat([df, normalized_df], axis=1)

        if "observed" in df.columns:
            df["observed_norm"] = self._y_scaler.transform(
                df["observed"].values.reshape(-1, 1)
            )

        return df

    def _add_temperature_bin_masked_ts(self, df):
        settings = self.settings.TEMPERATURE_BIN

        def get_k(int_col, a, b):
            k = []
            for hour in range(24):
                df_hour = df[df["hour_of_day"] == hour]
                df_hour = df_hour.sort_values(by=int_col)

                x_data = a*df_hour[int_col].values + b
                y_data = df_hour["observed"].values

                # Fit the model using robust least squares
                try:
                    params = fit_exp_growth_decay(x_data, y_data, k_only=True, is_x_sorted=True)
                    # save k for each hour
                    k.append(params[2])
                except:
                    pass

            k = np.abs(np.array(k))
            k = np.mean(k[k < 5])

            return k

        # TODO: if this permanent then it should not create, erase, make anew
        self._ts_feature_norm.remove("temperature_norm")

        # get all the temp_bin columns
        # get list of columns beginning with "daily_temp_" and ending in a number
        for interaction_col in ["temp_bin_", "temporal_cluster_"]:
            cols = [col for col in df.columns if col.startswith(interaction_col) and col[-1].isdigit()]
            for col in cols:
                # splits temperature_norm into unique columns if that temp_bin column is True
                ts_col = f"{col}_ts"
                df[ts_col] = df["temperature_norm"] * df[col]

                self._ts_feature_norm.append(ts_col)

        # TODO: if this is permanent then it should be a function, not this mess
        if settings.INCLUDE_EDGE_BINS:
            if self._T_edge_bin_coeffs is None:
                self._T_edge_bin_coeffs = {}

            cols = [col for col in df.columns if col.startswith("temp_bin_") and col[-1].isdigit()]
            cols = [0, int(cols[-1].replace("temp_bin_", ""))]
            # maybe add nonlinear terms to second and second to last columns?
            # cols = [0, 1, last_temp_bin - 1, last_temp_bin]
            # cols = list(set(cols))
            # all columns?
            # cols = range(cols[0], cols[1] + 1)

            for n in cols:
                base_col = f"temp_bin_{n}"
                int_col = f"{base_col}_ts"
                T_col = f"{base_col}_T"

                # get k for exponential growth/decay
                if not self.is_fit:
                    # determine temperature conversion for bin
                    range_offset = settings.EDGE_BIN_TEMPERATURE_RANGE_OFFSET
                    T_range = [df[int_col].min() - range_offset, df[int_col].max() + range_offset]
                    new_range = [-1, 1]

                    T_a = (new_range[1] - new_range[0])/(T_range[1] - T_range[0])
                    T_b = new_range[1] - T_a*T_range[1]

                    # The best rate for exponential
                    if settings.EDGE_BIN_RATE == "heuristic":
                        k = get_k(int_col, T_a, T_b)
                    else:
                        k = settings.EDGE_BIN_RATE

                    # get A for exponential
                    A = 1/(np.exp(1/k*new_range[1]) - 1)

                    self._T_edge_bin_coeffs[n] = {
                        "T_A": float(T_a),
                        "T_B": float(T_b),
                        "K": float(k),
                        "A": float(A),
                    }

                T_a = self._T_edge_bin_coeffs[n]["T_A"]
                T_b = self._T_edge_bin_coeffs[n]["T_B"]
                k = self._T_edge_bin_coeffs[n]["K"]
                A = self._T_edge_bin_coeffs[n]["A"]

                df[T_col] = np.where(
                    df[base_col].values,
                    T_a * df[int_col].values + T_b,
                    0
                )

                for pos_neg in ["pos", "neg"]:
                    # if first or last column, add additional column
                    # testing exp, previously squaring worked well

                    s = 1
                    if "neg" in pos_neg:
                        s = -1

                    # set rate exponential
                    ts_col = f"{base_col}_{pos_neg}_exp_ts"

                    df[ts_col] = np.where(
                        df[base_col].values,
                        A * np.exp(s / k * df[T_col].values) - A,
                        0
                    )

                    self._ts_feature_norm.append(ts_col)

        return df

    def _get_feature_matrices(self, df, dst_indices):
        # get aggregated features with agg function
        agg_dict = {f: lambda x: list(x) for f in self._ts_feature_norm}

        def correct_dst(agg):
            """interpolate or average hours to account for DST. modifies in place"""
            interp, mean = dst_indices
            for date, hour in interp:
                for feature_idx, feature in enumerate(agg[date]):
                    if hour == 0:
                        # there are a handful of countries that use 0:00 as the DST transition
                        interpolated = (
                            agg[date - 1][feature_idx][-1] + feature[hour]
                        ) / 2
                    else:
                        interpolated = (feature[hour - 1] + feature[hour]) / 2
                    feature.insert(hour, interpolated)
            for date, hour in mean:
                for feature in agg[date]:
                    mean = (feature[hour + 1] + feature.pop(hour)) / 2
                    feature[hour] = mean

        agg_x = df.groupby("date").agg(agg_dict).values.tolist()
        correct_dst(agg_x)

        # get the features and target for each day
        ts_feature = np.array(agg_x)

        ts_feature = ts_feature.reshape(
            ts_feature.shape[0], ts_feature.shape[1] * ts_feature.shape[2]
        )

        # get the first categorical features for each day for each sample
        unique_dummies = (
            df[self._categorical_features + ["date"]].groupby("date").first()
        )

        X = np.concatenate((ts_feature, unique_dummies), axis=1)

        if not self.is_fit:
            agg_y = (
                df.groupby("date")
                .agg({"observed_norm": lambda x: list(x)})
                .values.tolist()
            )
            correct_dst(agg_y)
            y = np.array(agg_y)
            y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

        else:
            y = None

        return X, y

    def to_dict(self):
        feature_scaler = {}
        if self.settings.SCALING_METHOD == _settings.ScalingChoice.STANDARDSCALER:
            for i, key in enumerate(self._ts_features):
                feature_scaler[key] = [
                    self._feature_scaler.mean_[i],
                    self._feature_scaler.scale_[i],
                ]

            y_scaler = [self._y_scaler.mean_, self._y_scaler.scale_]

        elif self.settings.SCALING_METHOD == _settings.ScalingChoice.ROBUSTSCALER:
            for i, key in enumerate(self._ts_features):
                feature_scaler[key] = [
                    self._feature_scaler.center_[i],
                    self._feature_scaler.scale_[i],
                ]

            y_scaler = [self._y_scaler.center_, self._y_scaler.scale_]

        # convert self._df_temporal_clusters to list of lists
        df_temporal_clusters = self._df_temporal_clusters.reset_index().values.tolist()

        params = _settings.SerializeModel(
            SETTINGS=self.settings,
            TEMPORAL_CLUSTERS=df_temporal_clusters,
            TEMPERATURE_BIN_EDGES=self._T_bin_edges,
            TEMPERATURE_EDGE_BIN_COEFFICIENTS=self._T_edge_bin_coeffs,
            TS_FEATURES=self._ts_features,
            CATEGORICAL_FEATURES=self._categorical_features,
            COEFFICIENTS=self._model.coef_.tolist(),
            INTERCEPT=self._model.intercept_.tolist(),
            FEATURE_SCALER=feature_scaler,
            CATAGORICAL_SCALER=None,
            Y_SCALER=y_scaler,
            BASELINE_METRICS=self.baseline_metrics,
        )

        return params.model_dump()

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        # get settings
        train_features = data.get("SETTINGS").get("TRAIN_FEATURES")

        if "ghi" in train_features:
            settings = _settings.HourlySolarSettings(**data.get("SETTINGS"))
        else:
            settings = _settings.HourlyNonSolarSettings(**data.get("SETTINGS"))

        # initialize model class
        model_cls = cls(settings=settings)

        df_temporal_clusters = pd.DataFrame(
            data.get("TEMPORAL_CLUSTERS"),
            columns=["day_of_week", "temporal_cluster"],
        ).set_index(["day_of_week"])

        model_cls._df_temporal_clusters = df_temporal_clusters
        model_cls._T_bin_edges = np.array(data.get("TEMPERATURE_BIN_EDGES"))
        model_cls._T_edge_bin_coeffs = {
            int(k): v for k, v in data.get("TEMPERATURE_EDGE_BIN_COEFFICIENTS").items()
        }

        model_cls._ts_features = data.get("TS_FEATURES")
        model_cls._categorical_features = data.get("CATEGORICAL_FEATURES")

        # set scalers
        feature_scaler_values = list(data.get("FEATURE_SCALER").values())
        feature_scaler_loc = [i[0] for i in feature_scaler_values]
        feature_scaler_scale = [i[1] for i in feature_scaler_values]

        y_scaler_values = data.get("Y_SCALER")

        if settings.SCALING_METHOD == _settings.ScalingChoice.STANDARDSCALER:
            model_cls._feature_scaler.mean_ = np.array(feature_scaler_loc)
            model_cls._feature_scaler.scale_ = np.array(feature_scaler_scale)

            model_cls._y_scaler.mean_ = np.array(y_scaler_values[0])
            model_cls._y_scaler.scale_ = np.array(y_scaler_values[1])

        elif settings.SCALING_METHOD == _settings.ScalingChoice.ROBUSTSCALER:
            model_cls._feature_scaler.center_ = np.array(feature_scaler_loc)
            model_cls._feature_scaler.scale_ = np.array(feature_scaler_scale)

            model_cls._y_scaler.center_ = np.array(y_scaler_values[0])
            model_cls._y_scaler.scale_ = np.array(y_scaler_values[1])

        # set model
        model_cls._model.coef_ = np.array(data.get("COEFFICIENTS"))
        model_cls._model.intercept_ = np.array(data.get("INTERCEPT"))

        model_cls.is_fit = True

        # set baseline metrics
        model_cls.baseline_metrics = BaselineMetricsFromDict(
            data.get("BASELINE_METRICS")
        )

        return model_cls

    @classmethod
    def from_json(cls, str_data):
        return cls.from_dict(json.loads(str_data))

    def plot(
        self,
        df_eval,
        ax=None,
        title=None,
        figsize=None,
        temp_range=None,
    ):
        """Plot a model fit.

        Parameters
        ----------
        ax : :any:`matplotlib.axes.Axes`, optional
            Existing axes to plot on.
        title : :any:`str`, optional
            Chart title.
        figsize : :any:`tuple`, optional
            (width, height) of chart.
        with_candidates : :any:`bool`
            If True, also plot candidate models.
        temp_range : :any:`tuple`, optionl
            Temperature range to plot

        Returns
        -------
        ax : :any:`matplotlib.axes.Axes`
            Matplotlib axes.
        """
        raise NotImplementedError