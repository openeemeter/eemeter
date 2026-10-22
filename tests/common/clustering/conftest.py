#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Fixtures shared by the clustering tests."""

import numpy as np
import pytest



@pytest.fixture
def simple_2d_data():
    """Three well-separated 10-d clusters, 50 points each (150 total), drawn from the
    global NumPy RNG seeded at 42."""
    np.random.seed(42)
    cluster1 = np.random.randn(50, 10) + np.array([0] * 10)
    cluster2 = np.random.randn(50, 10) + np.array([5] * 10)
    cluster3 = np.random.randn(50, 10) + np.array([10] * 10)
    return np.vstack([cluster1, cluster2, cluster3])
