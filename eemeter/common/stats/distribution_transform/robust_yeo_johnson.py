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

from scipy.stats import norm
from scipy.optimize import minimize_scalar

from eemeter.common.stats.distribution_transform.standardize import (
    robust_standardize,
)
from eemeter.common.stats.distribution_transform.mu_sigma import adaptive_weighted_mu_sigma, robust_mu_sigma
from eemeter.common.stats.adaptive_loss import adaptive_loss_fcn


# Work based on RayMaekers 2021 paper titled "Transforming variables to central normality"
# https://doi.org/10.1007/s10994-021-05960-5

# TODO: interesting article: https://link.springer.com/article/10.1007/s10260-022-00640-7


def _yeo_johnson_base(x, lam, deriv=False):
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
        if   (lam != 0) and (x >= 0):
            return (x + 1)**(lam - 1)
        elif (lam == 0) and (x >= 0):
            return 1/(1 + x)
        elif (lam != 2) and (x < 0):
            return (1 - x)**(1 - lam)
        elif (lam == 2) and (x < 0):
            return 1/(1 - x)


def _box_cox_base(x, lam, deriv=False):
    if not deriv:
        if   (lam != 0):
            return (x**lam - 1)/lam
        elif (lam == 0):
            return np.log(x)

    else:
        if   (lam != 0):
            return x**(lam - 1)
        elif (lam == 0):
            return 1/x


def rectified_transform(x, lam, Q=None, Q_perc=0.25, tr_type="Yeo-Johnson"):
    if tr_type == "Yeo-Johnson":
        tr = _yeo_johnson_base
    elif tr_type == "Box-Cox":
        tr = _box_cox_base

    if Q is None and Q_perc is not None:
        [q1, q3] = np.quantile(x, [Q_perc, 1 - Q_perc])
          
    elif Q is not None:
        [q1, q3] = Q

    h = np.empty_like(x)
    for i in range(len(x)):
        if Q is None:
            h[i] = tr(x[i], lam)

        else:
            if (q1 <= x[i]) and (x[i] < q3):
                h[i] = tr(x[i], lam)

            elif q3 < x[i]:
                h[i] = tr(q3, lam) + (x[i] - q3)*tr(q3, lam, deriv=True)

            elif x[i] < q1:
                h[i] = tr(q1, lam) + (x[i] - q1)*tr(q1, lam, deriv=True)

    return h


def unrectified_transform(x, lam, tr_type="Yeo-Johnson"):
    if tr_type == "Yeo-Johnson":
        tr = _yeo_johnson_base
    elif tr_type == "Box-Cox":
        tr = _box_cox_base

    h = np.empty_like(x)
    for i in range(len(x)):
        h[i] = tr(x[i], lam)

    return h


def loss_fcn(x, mu=0, c=1, loss_type="adaptive"):
    if loss_type == "adaptive":
        loss, _ = adaptive_loss_fcn(x, mu=mu, c=c, alpha="adaptive", replace_nonfinite=True)
        return loss
    
    elif loss_type == "tukey_bisquare":
        return np.piecewise(x, [np.abs(x) <= c, np.abs(x) > c], [lambda x: 1 - (1 - (x/c)**2)**3, 1])

    else:
        raise NotImplementedError(f"loss_type: {loss_type} not implemented")


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
    if robust_type == "huber_m_estimate":
        mu, sigma = robust_mu_sigma(x, robust_type, c=c_huber, tol=1e-08)
    else:
        mu, sigma = robust_mu_sigma(x, robust_type)

    h_0_standardized = np.abs((h_0 - mu)/sigma)

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
    Q=None, 
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
    bracket = np.array([-1, 1, 3])

    if pre_standardize:
        if transform_type == "Yeo-Johnson":
            mu, sigma = adaptive_weighted_mu_sigma(x, use_mean=False, rel_err=1E-4, abs_err=1E-4)
            x = (x - mu)/sigma

        elif transform_type == "Box-Cox":
            x = np.log(x)
            mu, sigma = adaptive_weighted_mu_sigma(x, use_mean=False, rel_err=1E-4, abs_err=1E-4)
            x = np.exp((x - mu)/sigma)

    x = np.sort(x)
    
    Q = np.quantile(x, [Q_perc, 1 - Q_perc])

    lam_loss_0 = initial_lam_obj_fcn_dec(x, Q, transform_type, c, robust_type, c_huber)
    # res = minimize_scalar(lam_loss_0, bounds=bounds, method="bounded")
    res = minimize_scalar(lam_loss_0, bracket=bracket[[0, 1]], method="brent")
    lam = res.x

    for _ in range(2):
        lam_loss = lam_obj_fcn_dec(x, lam, transform_type, robust_type, c_huber, outlier_alpha=outlier_alpha)

        # res = minimize_scalar(lam_loss, bounds=bounds, method="bounded")
        res = minimize_scalar(lam_loss, bracket=bracket[[0, 1]], method="brent")
        lam = res.x

    xt = rectified_transform(x, lam, Q=None, tr_type=transform_type)

    if post_standardize:
        if robust_type == "huber_m_estimate":
            xt = robust_standardize(xt, robust_type=robust_type, c=c_huber, tol=1e-08)
        else:
            xt = robust_standardize(xt, robust_type=robust_type)

    return xt, lam


def robust_YJ_transform(x, Q_perc=0.25, c=0.5, outlier_alpha=0.005, c_huber=1.5, robust_type="huber_m_estimate"):
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