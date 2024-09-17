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

import numpy as np
import pandas as pd

from eemeter.eemeter.models.hourly.model import HourlyModel, _get_dst_indices


# TODO: need to make explicit solar/nonsolar versions and set settings requirements/defaults appropriately
class HourlyTestModel(HourlyModel):
    def _prepare_features(self, meter_data):
        """
        Initializes the meter data by performing the following operations:
        - Renames the 'model' column to 'model_old' if it exists
        - Converts the index to a DatetimeIndex if it is not already
        - Adds a 'season' column based on the month of the index using the settings.season dictionary
        - Adds a 'day_of_week' column based on the day of the week of the index
        - Removes any rows with NaN values in the 'temperature' or 'observed' columns
        - Sorts the data by the index
        - Reorders the columns to have 'season' and 'day_of_week' first, followed by the remaining columns

        Parameters:
        - meter_data: A pandas DataFrame containing the meter data

        Returns:
        - A pandas DataFrame containing the initialized meter data
        """
        dst_indices = _get_dst_indices(meter_data)
        meter_data = self._add_categorical_features(meter_data)
        if self.settings.SUPPLEMENTAL_DATA is not None:
            self._add_supplemental_features(meter_data)

        self._sort_features()

        meter_data = self._daily_sufficiency(meter_data)
        meter_data = self._normalize_features(meter_data)
        meter_data = self._add_temperature_bin_ts(meter_data)

        X_predict, _ = self._get_feature_matrices(meter_data, dst_indices)

        if not self.is_fit:
            # remove insufficient days from fit data
            meter_data = meter_data[meter_data["include_date"]]
            X_fit, y_fit = self._get_feature_matrices(meter_data, dst_indices)

        else:
            X_fit, y_fit = None, None

        return X_fit, X_predict, y_fit
    

    def _add_temperature_bin_ts(self, df):
        extra_first_last = False

        settings = self.settings.TEMPERATURE_BIN

        # TODO: if this permanent then it should not create, erase, make anew
        self._ts_feature_norm.remove("temperature_norm")

        # get all the daily_temp columns
        # get list of columns beginning with "daily_temp_" and ending in a number
        for interaction_col in ["daily_temp_", "temporal_cluster_"]:
            cols = [col for col in df.columns if col.startswith(interaction_col) and col[-1].isdigit()]
            for col in cols:
                # splits temperature_norm into unique columns if that daily_temp column is True
                ts_col = f"{col}_ts"
                df[ts_col] = df["temperature_norm"] * df[col]

                self._ts_feature_norm.append(ts_col)

                # TODO: if this is permanent then it should be a function, not this mess
                if extra_first_last and (interaction_col == "daily_temp_"):
                    k = settings.EDGE_BIN_RATE
                    # if first or last column, add additional column
                    # testing exp, previously squaring worked well
                    if col == cols[0]:
                        # get indices where df[col] is True
                        idx = df.index[df[col]].values
                        temp_norm = df["temperature_norm"].values[idx]

                        # set neg rate exponential
                        name = f"daily_temp_0_neg_exp_ts"
                        df[name] = 0.0

                        df[name].iloc[idx] = 1/10*np.exp(-1/k*temp_norm)
                        self._ts_feature_norm.append(name)

                        # set pos rate exponential
                        name = f"daily_temp_0_pos_exp_ts"
                        df[name] = 0.0
                        
                        df[name].iloc[idx] = 1/10*np.exp(1/k*temp_norm)
                        self._ts_feature_norm.append(name)

                    elif col == cols[-1]:
                        idx = df.index[df[col]].values
                        temp_norm = df["temperature_norm"].values[idx]

                        # set neg rate exponential
                        i = col.replace("daily_temp_", "")
                        name = f"daily_temp_{i}_neg_exp_ts"
                        df[name] = 0.0

                        df[name].iloc[idx] = 1/10*np.exp(-1/k*temp_norm)
                        self._ts_feature_norm.append(name)

                        # set pos rate exponential
                        i = col.replace("daily_temp_", "")
                        name = f"daily_temp_{i}_pos_exp_ts"
                        df[name] = 0.0

                        df[name].iloc[idx] = 1/10*np.exp(1/k*temp_norm)
                        self._ts_feature_norm.append(name)

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