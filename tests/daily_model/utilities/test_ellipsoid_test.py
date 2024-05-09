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
import pandas as pd

from eemeter.eemeter.models.daily.utilities.ellipsoid_test import (
    ellipsoid_intersection_test,
    ellipsoid_K_function,
    robust_confidence_ellipse,
    ellipsoid_split_filter,
)


def test_ellipsoid_intersection_test():
    # test case 1: ellipsoids intersect
    mu_A = np.array([0, 0])
    mu_B = np.array([1, 1])
    cov_A = np.array([[1, 0], [0, 1]])
    cov_B = np.array([[1, 0], [0, 1]])
    assert ellipsoid_intersection_test(mu_A, mu_B, cov_A, cov_B) == True

    # test case 2: ellipsoids do not intersect
    mu_A = np.array([0, 0])
    mu_B = np.array([3, 3])
    cov_A = np.array([[1, 0], [0, 1]])
    cov_B = np.array([[1, 0], [0, 1]])
    assert ellipsoid_intersection_test(mu_A, mu_B, cov_A, cov_B) == False

    # test case 3: ellipsoids intersect at a single point
    mu_A = np.array([0, 0])
    mu_B = np.array([1, 0])
    cov_A = np.array([[1, 0], [0, 1]])
    cov_B = np.array([[1, 0], [0, 1]])
    assert ellipsoid_intersection_test(mu_A, mu_B, cov_A, cov_B) == True


def test_ellipsoid_K_function():
    # test case 1: ss = 0.5 -> doesn't match?
    ss = 0.5
    lambdas = np.array([1, 2, 3])
    v_squared = np.array([1, 2, 3])
    assert np.isclose(ellipsoid_K_function(ss, lambdas, v_squared), 0.0417, atol=1e-3)

    # test case 2: ss = 0
    ss = 0
    lambdas = np.array([1, 2])
    v_squared = np.array([1, 2])
    assert np.isclose(ellipsoid_K_function(ss, lambdas, v_squared), 1)

    # test case 3: ss = 1
    ss = 1
    lambdas = np.array([1, 2])
    v_squared = np.array([1, 2])
    assert np.isclose(ellipsoid_K_function(ss, lambdas, v_squared), 1)


def test_robust_confidence_ellipse():
    # test case 1: no outliers
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    mu, cov, a, b, phi = robust_confidence_ellipse(x, y)
    assert np.allclose(mu, np.array([2, 5]))
    assert np.allclose(cov, np.array([[1.96, 0], [0, 0]]))
    assert np.isclose(a, 1.4)
    assert np.isclose(b, 0.0)
    assert np.isclose(phi, 0.0)

    # test case 2: with outliers
    x = np.array([1, 2, 3, 100])
    y = np.array([4, 5, 6, 200])
    mu, cov, a, b, phi = robust_confidence_ellipse(x, y)
    assert np.allclose(mu, np.array([26.5, 5.5]))
    assert np.allclose(
        cov,
        np.array([[4.70726667e03, 3.26666667e01], [3.26666667e01, 6.53333333e-01]]),
        atol=1e-3,
    )
    assert np.isclose(a, 68.61117534074863)
    assert np.isclose(b, 0.6531602874075392)
    assert np.isclose(phi, 0.006940142826197432)


def test_ellipsoid_split_filter():
    # TODO : Add more test cases which contain all seasons and weekday/weekend
    # Test case 1: Test with a small dataset
    meter = pd.DataFrame(
        {
            "season": [
                "summer",
                "summer",
                "summer",
                "shoulder",
                "shoulder",
                "shoulder",
                "winter",
                "winter",
                "winter",
            ],
            "day_of_week": [1, 2, 3, 4, 5, 6, 7, 1, 2],
            "temperature": [20, 25, 30, 15, 20, 25, 10, 5, 0],
            "observed": [10, 20, 30, 15, 25, 35, 5, 10, 15],
        }
    )
    expected_output = {
        "summer": True,
        "shoulder": True,
        "winter": True,
        "weekday_weekend": True,
    }
    assert ellipsoid_split_filter(meter) == expected_output
