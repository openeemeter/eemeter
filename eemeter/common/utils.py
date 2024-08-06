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

import gc
import sys
import psutil
import multiprocessing as mp
import numba
import numpy as np
from numba.extending import overload


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


def to_np_array(x):
    """
    This function converts the input value 'x' to a numpy array.

    Parameters:
    x [int, float, array]: The input value to be converted to a numpy array.

    Returns:
    numpy array: The converted numpy array.
    """
    if x is None:
        return None

    if not hasattr(x, "__len__"):
        x = [x]

    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # if ndim is 0 then convert to 1D array
    if x.ndim == 0:
        x = np.array([x])

    return np.array(x)


@numba.jit(nopython=True, cache=True)
def _OoM(x, method="round"):
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


def OoM(x, method="round"):
    if not hasattr(x, "__len__"):
        return _OoM(np.array([x]), method=method)[0]

    return _OoM(to_np_array(x), method=method)


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


def sigmoid(x, x0=0, k=1):
    # https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
    
    def _positive_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(x):
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)

        return exp / (exp + 1)

    x = (x - x0) / k

    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    res = np.empty_like(x, dtype=float)
    res[positive] = _positive_sigmoid(x[positive])
    res[negative] = _negative_sigmoid(x[negative])

    return res


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


def _execute_with_mp(fcn, args_list, use_mp=True):
    """Runs a function with multiprocessing if use_mp is True, otherwise runs
    the function without multiprocessing.

    Args:
        fcn (function): The function to run.
        args_list (iterable): The list of arguments to pass to the function.
        use_mp (bool): Whether to use multiprocessing.

    Returns:
        The result of the function.
    """

    if len(args_list) == 1:
        use_mp = False

    if use_mp:
        # get memory size of args_list in gb
        args_list_size = get_obj_size(args_list) / (1024.0 ** 3)

        # get amount of memory available in gb
        memory_available = psutil.virtual_memory().available / (1024.0 ** 3)

        print("args_list_size: ", args_list_size)
        print("memory_available: ", memory_available)
        
        # if args_list is too large, use imap
        with mp.Pool(processes=mp.cpu_count()) as mp_pool:
            if args_list_size * 2 > memory_available:
                result = list(mp_pool.imap(fcn, args_list))
            else:
                result = mp_pool.map(fcn, args_list)

    else:
        result = []
        for args in args_list:
            result.append(fcn(args))

    return result