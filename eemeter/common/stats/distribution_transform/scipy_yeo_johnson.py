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

from scipy.stats import yeojohnson

from eemeter.common.stats.distribution_transform import robust_standardize


def scipy_YJ_transform(x, robust_type="huber_m_estimate"):
    x_std, _ = yeojohnson(x, lmbda=None)
    x_std = robust_standardize(x_std, robust_type)

    return x_std