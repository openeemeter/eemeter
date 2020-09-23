#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2020 EESampling contributors

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

from .__version__ import *
from .model import StratifiedSampling
from .bins import ModelSamplingException
from .diagnostics import StratifiedSamplingDiagnostics
from .bin_selection import StratifiedSamplingBinSelector
from .synthetic_data import SyntheticMeter, SyntheticPopulation, SyntheticTreatmentPoolPopulation
