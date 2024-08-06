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

from eemeter.common.transformation import Bisymlog
import numpy as np

from scipy.optimize import minimize_scalar
from scipy.stats import skew
from statsmodels.stats.stattools import robust_skewness as robust_skew

from eemeter.common.stats.distribution_transform.standardize import robust_standardize
from eemeter.common.stats.outliers import IQR_outlier


def bisymlog(x, rescale_quantile=None):
    def obj_fcn(X):
        C = 10**X

        xt = Bisymlog(C=C, rescale_quantile=rescale_quantile).transform(x)
        xt = robust_standardize(xt, robust_type="adaptive_weighted", use_mean=False, rel_err=1E-4, abs_err=1E-4)

        xt_outliers = IQR_outlier(xt, sigma_threshold=3, quantile=0.05)
        xt = xt[(xt_outliers[0] < xt) & (xt < xt_outliers[1])]

        abs_skew = np.abs(skew(xt))

        return abs_skew

    bnds = [-14, 6]

    res = minimize_scalar(obj_fcn, bounds=bnds, method='bounded')
    C = 10**res.x

    xt = Bisymlog(C=C, rescale_quantile=rescale_quantile).transform(x)
    xt = robust_standardize(xt, robust_type="adaptive_weighted", use_mean=False, rel_err=1E-4, abs_err=1E-4)

    return xt