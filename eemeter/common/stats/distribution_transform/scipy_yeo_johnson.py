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

from scipy.optimize import minimize_scalar
from scipy.stats import yeojohnson
from statsmodels.stats.stattools import robust_skewness as robust_skew

from eemeter.common.stats.distribution_transform import robust_standardize


def scipy_YJ(x, robust_type="huber_m_estimate"):
    x_std, _ = yeojohnson(x, lmbda=None)
    x_std = robust_standardize(x_std, robust_type)

    return x_std


def obj_fcn_dec(x):
    def obj_fcn(X):
        xt = yeojohnson(x, lmbda=X)

        n = 1 # [0: standard_skew, 1: quartile skew, 2: mean-median difference, standardized by abs deviation, 3: mean-median diff, standardized by std dev]
        abs_skew = np.abs(robust_skew(xt))[n]

        return abs_skew
    return obj_fcn


def robust_scipy_YJ(x, robust_type="huber_m_estimate", method="trim", **kwargs):
    idx_finite = np.argwhere(np.isfinite(x)).flatten()

    if len(idx_finite) < 3:
        return x

    # pre standardize x
    # x_finite = x[idx_finite]
    x_finite = robust_standardize(x[idx_finite], robust_type)

    if method == "trim":
        trim_quantile = 0.1
        if "trim_quantile" in kwargs:
            trim_quantile = kwargs["trim_quantile"]

        x_bnds = np.quantile(x_finite, [trim_quantile, 1 - trim_quantile])

        # get idx of x that is within the bounds
        idx_trim = np.argwhere((x_finite >= x_bnds[0]) & (x_finite <= x_bnds[1])).flatten()

        if len(idx_trim) >= 3:
            _, lmbda = yeojohnson(x_finite[idx_trim], lmbda=None)
        else:
            lmbda = None

    elif method == "skew":
        bnds = [-1, 1]

        obj_fcn = obj_fcn_dec(x_finite)
        res = minimize_scalar(obj_fcn, bracket=bnds, method='brent')
        lmbda = res.x

    if lmbda is not None:
        try:
            x[idx_finite] = yeojohnson(x_finite, lmbda=lmbda)
        except:
            pass # if yeojohnson fails, return x as is
    
    # post standardize x
    x[idx_finite] = robust_standardize(x[idx_finite], robust_type)

    return x