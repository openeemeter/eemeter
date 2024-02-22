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
from eemeter.eemeter.models.billing.data import BillingBaselineData, BillingReportingData
from eemeter import load_sample
import numpy as np
import pandas as pd
import pytest

TEMPERATURE_SEED = 29
METER_SEED = 41
NUM_DAYS_IN_YEAR = 365

@pytest.fixture
def get_datetime_index(request):
    # Request = [frequency , is_timezone_aware]

    # Create a DateTimeIndex at given frequency and timezone if requested
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq=request.param[0], tz = 'US/Eastern' if request.param[1] else None)

    return datetime_index

@pytest.fixture
def get_datetime_index_half_hourly_with_timezone():
    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq='30T', tz = 'US/Eastern')

    return datetime_index

@pytest.fixture
def get_datetime_index_hourly_with_timezone():
    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq='H', tz = 'US/Eastern')

    return datetime_index

@pytest.fixture
def get_datetime_index_daily_with_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq='D', tz = 'US/Eastern')

    return datetime_index

@pytest.fixture
def get_datetime_index_monthly_with_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq='MS', tz = 'US/Eastern')

    return datetime_index

@pytest.fixture
def get_datetime_index_bimonthly_with_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq='2MS', tz = 'US/Eastern')

    return datetime_index

@pytest.fixture
def get_datetime_index_daily_without_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq='D')

    return datetime_index

@pytest.fixture
def get_temperature_data_half_hourly(get_datetime_index_half_hourly_with_timezone):
    datetime_index = get_datetime_index_half_hourly_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    return df

@pytest.fixture
def get_temperature_data_hourly(get_datetime_index_hourly_with_timezone):
    datetime_index = get_datetime_index_hourly_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    return df

@pytest.fixture
def get_temperature_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    return df

@pytest.fixture
def get_meter_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    np.random.seed(METER_SEED)
    # Create a 'meter_value' column with random data
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed': meter_value}, index=datetime_index)

    return df

@pytest.fixture
def get_meter_data_monthly(get_datetime_index_monthly_with_timezone):
    datetime_index = get_datetime_index_monthly_with_timezone

    np.random.seed(METER_SEED)
    # Create a 'meter_value' column with random data
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed': meter_value}, index=datetime_index)

    return df

@pytest.fixture
def get_meter_data_bimonthly(get_datetime_index_bimonthly_with_timezone):
    datetime_index = get_datetime_index_bimonthly_with_timezone

    np.random.seed(METER_SEED)
    # Create a 'meter_value' column with random data
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed': meter_value}, index=datetime_index)

    return df

# Check that a missing timezone raises a Value Error
@pytest.mark.parametrize('get_datetime_index', [['D', False]], indirect=True)
def test_billing_baseline_data_with_missing_timezone(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'meter' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    with pytest.raises(ValueError):
        cls = BillingBaselineData(df, is_electricity_data=True)

# Check that a missing datetime index and column raises a Value Error
def test_billing_baseline_data_with_missing_datetime_index_and_column():

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(NUM_DAYS_IN_YEAR)

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(NUM_DAYS_IN_YEAR)

    # Create the DataFrame
    df = pd.DataFrame(data={'meter' : meter_value, 'temperature': temperature_mean})

    with pytest.raises(ValueError):
        cls = BillingBaselineData(df, is_electricity_data=True)

@pytest.mark.parametrize('get_datetime_index', [['MS', True]], indirect=True)
def test_billing_baseline_data_with_monthly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    # Because one month is missing
    assert len(cls.df) == 335
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    # DQ because only 12 days worth of temperature data is available
    assert len(cls.disqualification) == 2
    assert [dq.qualified_name for dq in cls.disqualification] == ['eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data']

@pytest.mark.parametrize('get_datetime_index', [['2MS', True]], indirect=True)
def test_billing_baseline_data_with_bimonthly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    np.random.seed(METER_SEED)
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    # Because two months are missing
    assert len(cls.df) == 305
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    # DQ because only 6 days worth of temperature data is available
    assert len(cls.disqualification) == 3
    assert [dq.qualified_name for dq in cls.disqualification] == ['eemeter.caltrack_sufficiency_criteria.incorrect_number_of_total_days','eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data']

def test_billing_baseline_data_with_monthly_hourly_frequencies(get_meter_data_monthly, get_temperature_data_hourly):

     # Create a DataFrame with uneven frequency
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_monthly

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0

def test_billing_baseline_data_with_bimonthly_hourly_frequencies(get_meter_data_bimonthly, get_temperature_data_hourly):

     # Create a DataFrame with uneven frequency
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_bimonthly

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0

def test_billing_baseline_data_with_monthly_daily_frequencies(get_meter_data_monthly, get_temperature_data_daily):

     # Create a DataFrame with uneven frequency
    df = get_temperature_data_daily

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_monthly

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    assert len(cls.disqualification) == 0

def test_billing_baseline_data_with_bimonthly_daily_frequencies(get_meter_data_bimonthly, get_temperature_data_daily):

     # Create a DataFrame with uneven frequency
    df = get_temperature_data_daily

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_bimonthly

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    assert len(cls.disqualification) == 0

def test_billing_baseline_data_with_specific_hourly_input():
    meter, temperature, _ = load_sample('il-electricity-cdd-hdd-hourly')
    meter = meter[meter.index.year==2017]
    temperature = temperature[temperature.index.year==2017]
    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == ['eemeter.data_quality.utc_index', 'eemeter.caltrack_sufficiency_criteria.inferior_model_usage']
    assert len(cls.disqualification) == 0

def test_billing_baseline_data_with_specific_daily_input():
    meter, temperature, _ = load_sample('il-electricity-cdd-hdd-daily')
    meter = meter[meter.index.year==2017]
    temperature = temperature[temperature.index.year==2017]
    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == ['eemeter.data_quality.utc_index', 'eemeter.caltrack_sufficiency_criteria.inferior_model_usage']
    assert len(cls.disqualification) == 0

def test_billing_baseline_data_with_specific_missing_daily_input():
    meter, temperature, _ = load_sample('il-electricity-cdd-hdd-daily')
    meter = meter[meter.index.year==2017]
    # Set 1 month meter data to NaN
    meter.loc[meter.index.month == 4] = np.nan

    temperature = temperature[temperature.index.year==2017]
    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == ['eemeter.data_quality.utc_index', 'eemeter.caltrack_sufficiency_criteria.inferior_model_usage']
    assert len(cls.disqualification) == 0

def test_billing_baseline_data_with_specific_monthly_input():
    meter, temperature, _ = load_sample('il-electricity-cdd-hdd-billing_monthly')
    meter = meter[meter.index.year==2017]
    temperature = temperature[temperature.index.year==2017]
    cls = BillingBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 1
    assert [warning.qualified_name for warning in cls.warnings] == ['eemeter.data_quality.utc_index']
    assert len(cls.disqualification) == 1
    # Since the first 21 days are missing so the first month is missing
    assert [disqualification.qualified_name for disqualification in cls.disqualification] == ['eemeter.caltrack_sufficiency_criteria.missing_monthly_meter_data']

@pytest.mark.parametrize('get_datetime_index', [['30T', True], ['H', True]], indirect=True)
def test_billing_reporting_data_with_missing_half_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])
    
    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df[mask].sample(frac=0.6).index, 'temperature'] = np.nan

    cls = BillingReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.missing_high_frequency_temperature_data'
    assert len(cls.disqualification) == 3
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.missing_monthly_temperature_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)

@pytest.mark.parametrize('get_datetime_index', [['D', True]], indirect=True)
def test_billing_reporting_data_with_missing_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    np.random.seed(TEMPERATURE_SEED)
    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])
    
    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df[mask].sample(frac=0.6).index, 'temperature'] = np.nan

    cls = BillingReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == NUM_DAYS_IN_YEAR
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    assert len(cls.disqualification) == 3
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.missing_monthly_temperature_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)