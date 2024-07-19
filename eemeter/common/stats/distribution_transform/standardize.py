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

from eemeter.common.stats.distribution_transform.mu_sigma import robust_mu_sigma


def robust_standardize(x, robust_type="iqr", **kwargs):
    mu, sigma = robust_mu_sigma(x, robust_type, **kwargs)
    x_std = (x - mu)/sigma

    return x_std