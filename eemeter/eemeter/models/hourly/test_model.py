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

from scipy.spatial.distance import cdist

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler

from timeit import default_timer as timer

import json

from eemeter.eemeter.models.hourly.model import HourlyModel
from eemeter.eemeter.models.hourly import settings as _settings
from eemeter.eemeter.models.hourly import HourlyBaselineData, HourlyReportingData
from eemeter.common.clustering import (
    transform as _transform,
    bisect_k_means as _bisect_k_means,
    scoring as _scoring,
)
from eemeter.common.metrics import BaselineMetrics, BaselineMetricsFromDict


# TODO: need to make explicit solar/nonsolar versions and set settings requirements/defaults appropriately
class TestHourlyModel(HourlyModel):
    _priority_cols = {
        "ts": ["temporal_cluster", "temp_bin", "temperature", "ghi"],
        "cat": ["temporal_cluster", "temp_bin"],
    }

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
                        edge_bin_count = settings.EDGE_BIN_HOURS

                        # get 5th smallest and 5th largest temperatures
                        sorted_temp = np.sort(df["temperature"].unique())
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
                    new_range = [-4, 4]

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
                        "T_a": T_a,
                        "T_b": T_b,
                        "k": k,
                        "A": A,
                    }

                T_a = self._T_edge_bin_coeffs[n]["T_a"]
                T_b = self._T_edge_bin_coeffs[n]["T_b"]
                k = self._T_edge_bin_coeffs[n]["k"]
                A = self._T_edge_bin_coeffs[n]["A"]

                df[T_col] = 0
                df.loc[df[base_col], T_col] = T_a*df[int_col] + T_b # doing this multiple times in get_k and here

                for pos_neg in ["pos", "neg"]:
                    # if first or last column, add additional column
                    # testing exp, previously squaring worked well

                    s = 1
                    if "neg" in pos_neg:
                        s = -1

                    # set rate exponential
                    ts_col = f"{base_col}_{pos_neg}_exp_ts"

                    df[ts_col] = 0
                    df.loc[df[base_col], ts_col] = A*np.exp(s/k*df[T_col]) - A

                    self._ts_feature_norm.append(ts_col)

        return df
    
def fit_exp_growth_decay(x, y, k_only=True, is_x_sorted=False):
    # Courtsey: https://math.stackexchange.com/questions/1337601/fit-exponential-with-constant
    #           https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales
    #           Jean Jacquelin

    # fitting function is actual b*exp(c*x) + a

    # sort x in order
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    if not is_x_sorted:
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

    s = [0]
    for i in range(1, len(x)):
        s.append(s[i-1] + 0.5*(y[i] + y[i-1])*(x[i] - x[i-1]))

    s = np.array(s)
    
    x_diff_sq = np.sum((x - x[0])**2)
    xs_diff = np.sum(s*(x - x[0]))
    s_sq = np.sum(s**2)
    xy_diff = np.sum((x - x[0])*(y - y[0]))
    ys_diff = np.sum(s*(y - y[0]))

    A = np.array([[x_diff_sq, xs_diff], [xs_diff, s_sq]])
    b = np.array([xy_diff, ys_diff])

    _, c = np.linalg.solve(A, b)
    k = 1/c

    if k_only:
        a, b = None, None
    else:
        theta_i = np.exp(c*x)

        theta = np.sum(theta_i)
        theta_sq = np.sum(theta_i**2)
        y_sum = np.sum(y)
        y_theta = np.sum(y*theta_i)

        A = np.array([[n, theta], [theta, theta_sq]])
        b = np.array([y_sum, y_theta])

        a, b = np.linalg.solve(A, b)

    return a, b, k
