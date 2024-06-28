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

from copy import deepcopy as copy
import numpy as np
from statsmodels.robust.scale import Huber as huber_m_estimate

from eemeter.common.stats.adaptive_loss import adaptive_weights
from eemeter.common.stats.basic import (
    MAD_k, 
    weighted_quantile,
)


def adaptive_weighted_mu_sigma(x, use_mean=False, rel_err=1E-4, abs_err=1E-4):
    mu = np.median(x)
    sigma = np.median(np.abs(x - mu))*MAD_k

    for n in range(10):
        mu_prior = copy(mu)
        sigma_prior = copy(sigma)
        weight = adaptive_weights(x, mu=mu_prior, sigma=sigma_prior)[0]
        if use_mean:
            mu = np.sum(weight*x)/np.sum(weight)
            sigma = np.sum(weight*(x - mu)**2)/np.sum(weight)

        else:
            mu = weighted_quantile(x, 0.5, weights=weight)
            sigma = weighted_quantile(np.abs(x - mu), 0.5, weights=weight)*MAD_k

        max_abs_err = np.max(np.abs([(mu - mu_prior), (sigma - sigma_prior)]))
        max_rel_err = np.max(np.abs([(mu - mu_prior)/mu_prior, (sigma - sigma_prior)/sigma_prior]))

        if (max_rel_err < rel_err) | (max_abs_err < abs_err):
            break

    if sigma == 0:
        sigma = 1

    return mu, sigma


def robust_mu_sigma(x, robust_type="huber_m_estimate", **kwargs):
    if robust_type == "huber_m_estimate":
        if "maxiter" not in kwargs:
            kwargs["maxiter"] = 50

        mu, sigma = huber_m_estimate(**kwargs)(x)

    elif robust_type == "adaptive_weighted":
        mu, sigma = adaptive_weighted_mu_sigma(x, **kwargs)

    return mu, sigma