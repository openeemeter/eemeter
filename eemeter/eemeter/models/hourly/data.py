#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2023 OpenEEmeter contributors

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
import pandas as pd
import numpy as np
from eemeter import merge_features, compute_temperature_features, caltrack_sufficiency_criteria
from typing import Union, Optional

class HourlyBaseline:
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool):
        if is_electricity_data:
            df.loc[df['observed'] == 0, 'observed'] = np.nan
        self.df = df
        self.sufficiency_warnings = self._check_data_sufficiency()

    def _check_data_sufficiency(self):
        meter = self.df['observed'].rename('meter_value')
        temp = self.df['temperature']
        temperature_features = compute_temperature_features(
            meter.index,
            temp,
            data_quality=True,
        )
        sufficiency_df = merge_features([meter, temperature_features])
        sufficiency = caltrack_sufficiency_criteria(sufficiency_df, requested_start=None, requested_end=None)
        return sufficiency.warnings

    @classmethod
    def from_series(cls, meter_data: Union[pd.Series, pd.DataFrame], temperature_data: Union[pd.Series, pd.DataFrame], is_electricity_data: bool):
        df = merge_features([meter_data, temperature_data])
        df = df.rename({
            df.columns[0]: 'observed',
            df.columns[1]: 'temperature',
        }, axis=1)
        return cls(df, is_electricity_data)


class HourlyReporting:
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool):
        #TODO add missing observed column
        if is_electricity_data:
            df.loc[df['observed'] == 0, 'observed'] = np.nan
        self.df = df

    @classmethod
    def from_series(cls, meter_data: Optional[pd.Series], temperature_data: pd.Series, is_electricity_data: bool):
        #TODO verify
        if meter_data is None:
            meter_data = temperature_data.copy().rename('observed') * np.nan
        df = merge_features([meter_data, temperature_data], keep_partial_nan_rows=True)
        df = df.rename({
            df.columns[0]: 'observed',
            df.columns[1]: 'temperature',
        }, axis=1)
        return cls(df, is_electricity_data)