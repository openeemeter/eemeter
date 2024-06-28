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

from __future__ import annotations

from enum import Enum

# used in clustering settings
from eemeter.gridmeter._utils.const import DistanceMetric
from eemeter.gridmeter._utils.const import AggType


class ScoreChoice(str, Enum):
    SILHOUETTE = "silhouette"
    SILHOUETTE_MEDIAN = "silhouette_median"
    VARIANCE_RATIO = "variance_ratio"
    CALINSKI_HARABASZ = "calinski-harabasz"
    DAVIES_BOULDIN = "davies-bouldin"


class NormalizeChoice(str, Enum):
    MIN_MAX = "min_max"
    NONE = "none"