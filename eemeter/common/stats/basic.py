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

import numba
import numpy as np

from scipy.stats import norm as norm_dist
from scipy.stats import t as t_dist

from eemeter.common.utils import to_np_array


MAD_k = 1 / norm_dist.ppf(0.75)  # Conversion factor from MAD to std for normal distribution


def t_stat(alpha, n, tail=2):
    """
    Calculate the t-statistic for a given alpha level, sample size, and tail type.

    Parameters:
    alpha (float): The significance level.
    n (int): The sample size.
    tail (int or str): The type of tail test. Can be 1 or "one" for one-tailed test,
                       and 2 or "two" for two-tailed test. Default is 2.

    Returns:
    float: The calculated t-statistic.
    """

    degrees_of_freedom = n - 1
    if tail == "one" or tail == 1:
        perc = 1 - alpha
    elif tail == "two" or tail == 2:
        perc = 1 - alpha / 2

    return t_dist.ppf(perc, degrees_of_freedom, 0, 1)


def z_stat(alpha, tail=2):
    if (tail == "one") or (tail == 1):
        perc = 1 - alpha
    elif (tail == "two") or (tail == 2):
        perc = 1 - alpha / 2
    else:
        raise Exception("incorrect type of tail input to 't_stat'")

    return norm_dist.ppf(np.asarray(perc), 0, 1)


def unc_factor(n, interval="PI", alpha=0.10):
    """
    Calculates the uncertainty factor for a given sample size, confidence interval type, and significance level.

    Parameters:
    n (int): The sample size.
    interval (str, optional): The type of confidence interval. Defaults to "PI" (Prediction Interval).
    alpha (float, optional): The significance level. Defaults to 0.10.

    Returns:
    float: The uncertainty factor.
    """

    if interval == "CI":
        return t_stat(alpha, n) / np.sqrt(n)

    if interval == "PI":
        return t_stat(alpha, n) * (1 + 1 / np.sqrt(n))


def median_absolute_deviation(x, mu=None, weights=None):
    """
    This function calculates the Median Absolute Deviation (MAD) of a given array.

    Parameters:
    x (numpy array): The input array for which the MAD is to be calculated.

    Returns:
    float: The calculated MAD of the input array.
    """

    if mu is None:
        mu = np.median(x)

    sigma = weighted_quantile(np.abs(x - mu), 0.5, weights=weights)*MAD_k

    return sigma


@numba.jit(nopython=True, cache=True)
def weighted_std(x, w, mean=None, w_sum_err=1e-6):
    """
    Calculate the weighted standard deviation of a given array.

    Parameters:
    x (numpy.ndarray): The input array.
    w (numpy.ndarray): The weights for the input array.
    mean (float, optional): The mean value. If None, the mean is calculated from the input array. Defaults to None.
    w_sum_err (float, optional): The error tolerance for the sum of weights. Defaults to 1e-6.

    Returns:
    float: The calculated weighted standard deviation.
    """

    n = float(len(x))

    w_sum = np.sum(w)
    if w_sum < 1 - w_sum_err or w_sum > 1 + w_sum_err:
        w /= w_sum

    if mean is None:
        mean = np.sum(w * x)

    var = np.sum(w * np.power((x - mean), 2)) / (1 - 1 / n)

    return np.sqrt(var)


def fast_std(x, weights=None, mean=None):
    """
    Function to calculate the approximate standard deviation quickly of a given array.
    This function can handle both weighted and unweighted calculations.

    Parameters:
    x (numpy.ndarray): The input array for which the standard deviation is to be calculated.
    weights (numpy.ndarray, optional): An array of weights for the input array. Defaults to None.
    mean (float, optional): The mean of the input array. If not provided, it will be calculated. Defaults to None.

    Returns:
    float: The calculated standard deviation.
    """

    if isinstance(weights, int) or isinstance(weights, float):
        weights = np.array([weights])

    if weights is None or len(weights) == 1 or np.allclose(weights - weights[0], 0):
        if mean is None:
            return np.std(x)

        else:
            n = float(len(x))
            var = np.sum(np.power((x - mean), 2)) / n
            return np.sqrt(var)

    else:
        if mean is None:
            mean = np.average(x, weights=weights)

        return weighted_std(x, weights, mean)


@numba.jit(nopython=True, cache=True)
def _weighted_quantile(
    values, 
    quantiles, 
    weights=None, 
    values_presorted=False, 
    old_style=False,
):
    """https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_presorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.quantile.
    :return: numpy.array with computed quantiles.
    """
    for q in quantiles:
        if not 0 <= q <= 1:
            raise ValueError("quantiles should be in [0, 1]")
        
    finite_idx = np.where(np.isfinite(values))
    values = values[finite_idx]

    if weights is None: # or (weights.size == 0):
        weights = np.ones_like(values)
    else:
        weights = weights[finite_idx]

    if not values_presorted:
        sorted_idx = np.argsort(values)
        values = values[sorted_idx]
        weights = weights[sorted_idx]

    res = np.cumsum(weights) - 0.5 * weights
    if old_style:  # To be convenient with numpy.quantile
        res -= res[0]
        res /= res[-1]
    else:
        res /= np.sum(weights)

    return np.interp(quantiles, res, values)


def weighted_quantile(
    values, 
    quantiles, 
    weights=None, 
    values_presorted=False, 
    old_style=False,
):
    values = to_np_array(values)
    quantiles = to_np_array(quantiles)

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = to_np_array(weights)

    try:
        res = _weighted_quantile(
            values, 
            quantiles, 
            weights, 
            values_presorted, 
            old_style
        )
    except:
        print(values)
        print(quantiles)
        print(weights)

        raise Exception("Error in weighted_quantile")

    return res