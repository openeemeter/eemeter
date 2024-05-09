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

SufficiencyRequirements = {}


class Settings:
    def __init__(self):
        pass


class DailySettings(Settings):
    def __init__(self, n_days_kept_min: int = 350, cvrmse_adj_max: float = 0.3):
        # TODO : Reuse the daily settings for Caltrack at eemeter/eemeter/caltrack/daily/utilities/config.py
        self.n_days_kept_min = n_days_kept_min
        self.cvrmse_adj_max = cvrmse_adj_max


class MonthlySettings(Settings):
    def __init__(self, n_months_kept_min: int = 12, cvrmse_adj_max: float = 0.3):
        self.n_months_kept_min = n_months_kept_min
        self.cvrmse_adj_max = cvrmse_adj_max
