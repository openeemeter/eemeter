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
import numba
from numba import float64, boolean

from scipy.stats import norm
from scipy.optimize import minimize_scalar

from eemeter.common.stats.distribution_transform.standardize import (
    robust_standardize,
)
from eemeter.common.stats.distribution_transform.mu_sigma import adaptive_weighted_mu_sigma, robust_mu_sigma
from eemeter.common.stats.adaptive_loss import adaptive_loss_fcn


# Work based on RayMaekers 2021 paper titled "Transforming variables to central normality"
# https://doi.org/10.1007/s10994-021-05960-5

# TODO: interesting article: https://link.springer.com/article/10.1007/s10260-022-00640-7#Sec17
#                            https://github.com/UniprJRC/FSDA/tree/master/toolbox/regression
#                            https://github.com/UniprJRC/FSDA/blob/master/toolbox/regression/FSRfan.m
#                            https://github.com/UniprJRC/FSDA/blob/master/toolbox/regression/fanBIC.m

_NO_DERIV = False
_DERIV = True


@numba.jit((float64)(float64, float64, boolean), nopython=True, error_model="numpy", cache=True)
def _yeo_johnson_base(x, lam, deriv):
    if not deriv:
        if   (lam != 0) and (x >= 0):
            return ((1 + x)**lam - 1)/lam
        elif (lam == 0) and (x >= 0):
            return np.log(1 + x)
        elif (lam != 2) and (x < 0):
            return -((1 - x)**(2 - lam) - 1)/(2 - lam)
        elif (lam == 2) and (x < 0):
            return -np.log(1 - x)
        else:
            return np.nan

    else:
        if   (lam != 0) and (x >= 0):
            return (x + 1)**(lam - 1)
        elif (lam == 0) and (x >= 0):
            return 1/(1 + x)
        elif (lam != 2) and (x < 0):
            return (1 - x)**(1 - lam)
        elif (lam == 2) and (x < 0):
            return 1/(1 - x)
        else:
            return np.nan


@numba.jit((float64)(float64, float64, boolean), nopython=True, error_model="numpy", cache=True)
def _box_cox_base(x, lam, deriv):
    if not deriv:
        if   (lam != 0):
            return (x**lam - 1)/lam
        elif (lam == 0):
            return np.log(x)
        else:
            return np.nan

    else:
        if   (lam != 0):
            return x**(lam - 1)
        elif (lam == 0):
            return 1/x
        else:
            return np.nan


@numba.jit(nopython=True, error_model="numpy", cache=True)
def rectified_transform(x, lam, Q, tr_type="Yeo-Johnson"):
    if tr_type == "Yeo-Johnson":
        tr = _yeo_johnson_base
    elif tr_type == "Box-Cox":
        tr = _box_cox_base

    [q1, q3] = Q

    h = np.empty_like(x)
    for i, xi in enumerate(x):
        if (q1 <= xi) and (xi < q3):
            h[i] = tr(xi, lam, _NO_DERIV)

        elif q3 < xi:
            h[i] = tr(q3, lam, _NO_DERIV) + (xi - q3)*tr(q3, lam, _DERIV)

        elif xi < q1:
            h[i] = tr(q1, lam, _NO_DERIV) + (xi - q1)*tr(q1, lam, _DERIV)

    return h


@numba.jit(nopython=True, error_model="numpy", cache=True)
def unrectified_transform(x, lam, tr_type="Yeo-Johnson"):
    if tr_type == "Yeo-Johnson":
        tr = _yeo_johnson_base
    elif tr_type == "Box-Cox":
        tr = _box_cox_base

    h = np.empty_like(x)
    for i, xi in enumerate(x):
        h[i] = tr(xi, lam, _NO_DERIV)

    return h


def loss_fcn(x, mu=0, c=1, loss_type="adaptive"):
    if loss_type == "adaptive":
        loss, _ = adaptive_loss_fcn(x, mu=mu, c=c, alpha="adaptive", replace_nonfinite=True)
        return loss
    
    elif loss_type == "tukey_bisquare":
        return np.piecewise(x, [np.abs(x) <= c, np.abs(x) > c], [lambda x: 1 - (1 - (x/c)**2)**3, 1])

    else:
        raise NotImplementedError(f"loss_type: {loss_type} not implemented")


def _robust_standardize(x, robust_type, c_huber):
    if robust_type == "huber_m_estimate":
        return robust_standardize(x, robust_type=robust_type, c=c_huber, tol=1e-08)
    else:
        return robust_standardize(x, robust_type=robust_type)


def initial_lam_obj_fcn_dec(x, Q, transform_type="Yeo-Johnson", c=0.5, robust_type="huber_m_estimate", c_huber=1.5):
    phi = norm.ppf((np.arange(0, len(x)) + 2/3)/(len(x) + 1/3))

    if robust_type == "huber_m_estimate":
        mu, sigma = robust_mu_sigma(x, robust_type, c=c_huber, tol=1e-08)
    else:
        mu, sigma = robust_mu_sigma(x, robust_type)

    def lam_obj_fcn(lam):
        h = rectified_transform(x, lam, Q, tr_type=transform_type)
        
        loss = loss_fcn((h - mu)/sigma - phi, mu=0, c=c, loss_type="tukey_bisquare")
        # loss = loss_fcn((h - mu)/sigma - phi, mu=0, c=c, loss_type="adaptive")

        return np.sum(loss)

    return lam_obj_fcn


def lam_obj_fcn_dec(
    x, 
    lam_0, 
    transform_type="Yeo-Johnson",
    robust_type="huber_m_estimate",
    c_huber=1.5, 
    outlier_alpha=0.005, 
):
    h_0 = unrectified_transform(x, lam_0, tr_type=transform_type)
    h_0_standardized = np.abs(_robust_standardize(h_0, robust_type, c_huber))

    phi = norm.ppf(1 - outlier_alpha)

    weight = np.zeros_like(x)
    weight[h_0_standardized <= phi] = 1

    def lam_obj_fcn(lam):
        h = unrectified_transform(x, lam, tr_type=transform_type)
        if not np.any(weight):
            weighted_var = 1
        else:
            weighted_mu = np.sum(weight*h)/np.sum(weight)
            weighted_var = np.sum(weight*(h - weighted_mu)**2)/np.sum(weight)

        if transform_type == "Yeo-Johnson":
            ML = -0.5*np.log(weighted_var) + (lam - 1)*np.sign(x)*np.log(1 + np.abs(x))

        elif transform_type == "Box-Cox":
            ML = -0.5*np.log(weighted_var) + (lam - 1)*np.log(x)

        return -np.sum(weight*ML)

    return lam_obj_fcn


def normal_transformation(
    x, 
    Q_perc=0.25, 
    transform_type="Yeo-Johnson", 
    c=0.5,
    outlier_alpha=0.005, 
    c_huber=1.5, 
    robust_type="huber_m_estimate",
    pre_standardize=True,
    post_standardize=True,
    ):

    # bounds = np.array([-1, 3]) + np.array([-10, 10])
    # bracket = np.array([-1, 1, 3])
    lmbda_bnds = np.array([-10, 10])

    if pre_standardize:
        if transform_type == "Yeo-Johnson":
            x = _robust_standardize(x, robust_type, c_huber)
        elif transform_type == "Box-Cox":
            x = np.exp(_robust_standardize(np.log(x), robust_type, c_huber))

    x = np.sort(x)
    Q = np.quantile(x, [Q_perc, 1 - Q_perc])
    for n in range(3):
        if n == 0:
            lam_loss_0 = initial_lam_obj_fcn_dec(x, Q, transform_type, c, robust_type, c_huber)
            # res = minimize_scalar(lam_loss_0, bounds=lmbda_bnds, method="bounded")
            res = minimize_scalar(lam_loss_0, bracket=lmbda_bnds, method="brent")
            lam = res.x

        else:
            lam_loss = lam_obj_fcn_dec(x, lam, transform_type, robust_type, c_huber, outlier_alpha=outlier_alpha)
            # res = minimize_scalar(lam_loss, bounds=lmbda_bnds, method="bounded")
            res = minimize_scalar(lam_loss, bracket=lmbda_bnds, method="brent")
            lam = res.x

    xt = rectified_transform(x, lam, Q=Q, tr_type=transform_type)

    if post_standardize:
        xt = _robust_standardize(xt, robust_type, c_huber)

    return xt, lam


def raymaekers_robust_YJ(x, Q_perc=0.25, c=0.5, outlier_alpha=0.005, c_huber=1.5, robust_type="huber_m_estimate"):
    # outlier_alpha should be between 0.005 and 0.025 (0.005 is higher efficiency, less robust)

    if np.all(x == x[0]): # if all values are the same, do not transform, return
        return np.zeros_like(x)

    idx_finite = np.argwhere(np.isfinite(x)).flatten()
    idx_nonfinite = np.array([i for i in np.arange(len(x)) if i not in idx_finite])

    xt_yj_out = np.empty_like(x)    
    if len(idx_finite) > 3:
        xt_yj, _ = normal_transformation(
            x[idx_finite], 
            Q_perc=Q_perc, 
            transform_type="Yeo-Johnson", 
            c=c, 
            c_huber=c_huber, 
            outlier_alpha=outlier_alpha,
            robust_type=robust_type,
        )

        xt_yj_out[idx_finite] = xt_yj

    if len(idx_nonfinite) > 0:
        xt_yj_out[idx_nonfinite] = x[idx_nonfinite]
    
    return xt_yj_out