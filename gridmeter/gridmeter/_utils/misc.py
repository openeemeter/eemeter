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

from scipy.stats import t as t_dist
from scipy.stats import norm as norm_dist

import numba
from numba.extending import overload


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


# Yield successive n-sized chunks from list
def chunks(lst, n):
    for i in range(0, len(lst), n):
        list_chunk = lst[i : i + n]
        list_idx_chunk = np.arange(i, i + len(list_chunk))
        yield list_idx_chunk, list_chunk


@overload(np.clip)
def np_clip(a, a_min, a_max):
    """
    This function applies a clip operation on the input array 'a' using the provided minimum and maximum values.
    The clip operation ensures that all elements in 'a' are within the range [a_min, a_max].
    If an element in 'a' is less than 'a_min', it is replaced with 'a_min'.
    If an element in 'a' is greater than 'a_max', it is replaced with 'a_max'.
    NaN values in 'a' are preserved as NaN.

    Parameters:
    a (numpy array): The input array to be clipped.
    a_min (float): The minimum value for the clip operation.
    a_max (float): The maximum value for the clip operation.

    Returns:
    numpy array: The clipped array.
    """

    @numba.vectorize
    def _clip(a, a_min, a_max):
        """
        This is a vectorized implementation of the clip function.
        It applies the clip operation on each element of the input array 'a'.

        Parameters:
        a (float): The input value to be clipped.
        a_min (float): The minimum value for the clip operation.
        a_max (float): The maximum value for the clip operation.

        Returns:
        float: The clipped value.
        """

        if np.isnan(a):
            return np.nan
        elif a < a_min:
            return a_min
        elif a > a_max:
            return a_max
        else:
            return a

    def clip_impl(a, a_min, a_max):
        """
        This is a numba implementation of the clip function.
        It applies the clip operation on the input array 'a' using the provided minimum and maximum values.

        Parameters:
        a (numpy array): The input array to be clipped.
        a_min (float): The minimum value for the clip operation.
        a_max (float): The maximum value for the clip operation.

        Returns:
        numpy array: The clipped array.
        """

        return _clip(a, a_min, a_max)

    return clip_impl


def OoM(x, method="round"):
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    return OoM_numba(x, method=method)


@numba.jit(nopython=True, cache=numba_cache)
def OoM_numba(x, method="round"):
    """
    This function calculates the order of magnitude (OoM) of each element in the input array 'x' using the specified method.

    Parameters:
    x (numpy array): The input array for which the OoM is to be calculated.
    method (str): The method to be used for calculating the OoM. It can be one of the following:
                  "round" - round to the nearest integer (default)
                  "floor" - round down to the nearest integer
                  "ceil" - round up to the nearest integer
                  "exact" - return the exact OoM without rounding

    Returns:
    x_OoM (numpy array): The array of the same shape as 'x' containing the OoM of each element in 'x'.
    """

    x_OoM = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi == 0.0:
            x_OoM[i] = 1.0

        elif method.lower() == "floor":
            x_OoM[i] = np.floor(np.log10(np.abs(xi)))

        elif method.lower() == "ceil":
            x_OoM[i] = np.ceil(np.log10(np.abs(xi)))

        elif method.lower() == "round":
            x_OoM[i] = np.round(np.log10(np.abs(xi)))

        else:  # "exact"
            x_OoM[i] = np.log10(np.abs(xi))

    return x_OoM


def RoundToSigFigs(x, p):
    """
    This function rounds the input array 'x' to 'p' significant figures.

    Parameters:
    x (numpy.ndarray): The input array to be rounded.
    p (int): The number of significant figures to round to.

    Returns:
    numpy.ndarray: The rounded array.
    """

    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - OoM(x_positive))
    return np.round(x * mags) / mags


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
    if (tail == "one") or (tail == 1):
        perc = 1 - alpha
    elif (tail == "two") or (tail == 2):
        perc = 1 - alpha / 2
    else:
        raise Exception("incorrect type of tail input to 't_stat'")

    return t_dist.ppf(perc, degrees_of_freedom, 0, 1)


def z_stat(alpha, tail=2):
    if (tail == "one") or (tail == 1):
        perc = 1 - alpha
    elif (tail == "two") or (tail == 2):
        perc = 1 - alpha / 2
    else:
        raise Exception("incorrect type of tail input to 't_stat'")

    return norm_dist.ppf(np.asarray(perc), 0, 1)


def unc_factor(n, interval="PI", alpha=0.05):
    """
    Calculates the uncertainty factor for a given sample size, confidence interval type, and significance level.

    Parameters:
    n (int): The sample size.
    interval (str, optional): The type of confidence interval. Defaults to "PI" (Prediction Interval).
    alpha (float, optional): The significance level. Defaults to 0.05.

    Returns:
    float: The uncertainty factor.
    """

    if interval == "CI":
        return t_stat(alpha, n) / np.sqrt(n)

    if interval == "PI":
        return t_stat(alpha, n) * (1 + 1 / np.sqrt(n))
