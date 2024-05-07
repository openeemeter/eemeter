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
from eemeter.eemeter.models.hourly.data import HourlyBaselineData, HourlyReportingData
from eemeter.eemeter.samples import load_sample
import numpy as np
import pandas as pd
import pytest

TEMPERATURE_SEED = 29
METER_SEED = 41
NUM_HOURS_IN_YEAR = 8760

@pytest.fixture
def get_datetime_index(request):
    # Request = [frequency , is_timezone_aware]

    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq=request.param[0], tz = 'US/Eastern' if request.param[1] else None)

    return datetime_index


# Check that a missing timezone raises a Value Error
@pytest.mark.parametrize('get_datetime_index', [['H', False]], indirect=True)
def test_hourly_baseline_data_with_missing_timezone(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    with pytest.raises(ValueError):
        cls = HourlyBaselineData(df, is_electricity_data=True)

# Check that a missing datetime index and column raises a Value Error
def test_hourly_baseline_data_with_missing_datetime_index_and_column():

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(NUM_HOURS_IN_YEAR)

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(NUM_HOURS_IN_YEAR)

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean})

    with pytest.raises(ValueError):
        cls = HourlyBaselineData(df, is_electricity_data=True)

@pytest.mark.parametrize('get_datetime_index', [['15T', True],['30T', True],['H', True]], indirect=True)
def test_daily_baseline_data_with_same_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = HourlyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_HOURS_IN_YEAR
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0

def test_daily_baseline_data_with_specific_hourly_input():
    meter, temperature, _ = load_sample('il-electricity-cdd-hdd-hourly')
    meter = meter[meter.index.year==2017]
    temperature = temperature[temperature.index.year==2017]
    cls = HourlyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_HOURS_IN_YEAR
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == ['eemeter.data_quality.utc_index', 'eemeter.sufficiency_criteria.extreme_values_detected']
    assert len(cls.disqualification) == 0


def test_duplicate_datetime_index_values():
    # Create a Timestamp with a specific date
    timestamp = pd.Timestamp('2023-01-01')

    # Create an Index with 365 identical timestamps
    datetime_index = pd.DatetimeIndex([timestamp]*8760)

    # Create random values for 'observed' and 'temperature'
    observed = np.random.rand(len(datetime_index))
    temperature = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed': observed, 'temperature': temperature}, index=datetime_index)
    
    cls = HourlyBaselineData(df, is_electricity_data=True)
    
    assert cls.df is not None
    assert(len(cls.df) == 1)


@pytest.mark.parametrize('get_datetime_index', [['15T', True], ['30T', True],['H', True]], indirect=True)
def test_hourly_reporting_data_with_half_hourly_and_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    cls = HourlyReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_HOURS_IN_YEAR
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0