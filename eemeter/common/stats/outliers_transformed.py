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

from eemeter.common.stats.distribution_transform import (
    bisymlog,
    scipy_YJ,
    robust_scipy_YJ,
    raymaekers_robust_YJ,
    robust_standardize,
)
from eemeter.common.stats.outliers import remove_outliers as basic_remove_outliers


def remove_outliers(x, weights=None, sigma_threshold=3, quantile=0.25, transform=None):
    if transform is None:
        xt = x
    elif transform == "standardize":
        xt = robust_standardize(x)
    elif transform == "bisymlog":
        xt = bisymlog(x)
    elif transform == "scipy_YJ":
        xt = scipy_YJ(x)
    elif transform == "robust_scipy_YJ":
        xt = robust_scipy_YJ(x)
    elif transform == "robust_YJ":
        xt = raymaekers_robust_YJ(x)

    _, idx_no_outliers = basic_remove_outliers(xt, weights, sigma_threshold, quantile)

    if len(idx_no_outliers) == 0:
        return x, []
    
    x_no_outliers = x[idx_no_outliers]

    return x_no_outliers, idx_no_outliers