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
        if settings.INCLUDE_EDGE_BINS:
            if self._T_edge_bin_coeffs is None:
                self._T_edge_bin_coeffs = {}

            cols = [col for col in df.columns if col.startswith("daily_temp_") and col[-1].isdigit()]
            cols = [0, int(cols[-1].replace("daily_temp_", ""))]
            cols = range(cols[0], cols[1] + 1)
            
            for n in cols:
                base_col = f"daily_temp_{n}"
                int_col = f"{base_col}_ts"
                T_col = f"{base_col}_T"

                # get k for exponential growth/decay
                if not self.is_fit:
                    # determine temperature conversion for bin
                    range_offset = settings.EDGE_BIN_TEMPERATURE_RANGE_OFFSET
                    T_range = [df[int_col].min() - range_offset, df[int_col].max() + range_offset]
                    new_range = [-3, 3]

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