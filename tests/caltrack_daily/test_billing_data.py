from eemeter.eemeter.models.billing.data import BillingBaselineData, BillingReportingData

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def get_datetime_index(request):
    # Request = [frequency , is_timezone_aware]

    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', freq=request.param[0])

    # Localize the DateTimeIndex to a timezone
    if request.param[1]:
        datetime_index = datetime_index.tz_localize('UTC')

    return datetime_index

@pytest.fixture
def get_datetime_index_half_hourly_with_timezone():
    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', freq='30T')

    # Localize the DateTimeIndex to a timezone
    datetime_index = datetime_index.tz_localize('UTC')

    return datetime_index

@pytest.fixture
def get_datetime_index_hourly_with_timezone():
    # Create a DateTimeIndex at 30-minute intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', freq='H')

    # Localize the DateTimeIndex to a timezone
    datetime_index = datetime_index.tz_localize('UTC')

    return datetime_index

@pytest.fixture
def get_datetime_index_daily_with_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    # Localize the DateTimeIndex to a timezone
    datetime_index = datetime_index.tz_localize('UTC')

    return datetime_index

@pytest.fixture
def get_datetime_index_daily_without_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')

    return datetime_index

@pytest.fixture
def get_temperature_data_half_hourly(get_datetime_index_half_hourly_with_timezone):
    datetime_index = get_datetime_index_half_hourly_with_timezone

    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    return df

@pytest.fixture
def get_temperature_data_hourly(get_datetime_index_hourly_with_timezone):
    datetime_index = get_datetime_index_hourly_with_timezone

    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    return df

@pytest.fixture
def get_meter_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    # Create a 'meter_value' column with random data
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed': meter_value}, index=datetime_index)

    return df

# Check that a missing timezone raises a Value Error
@pytest.mark.parametrize('get_datetime_index', [['D', False]], indirect=True)
def test_billing_baseline_data_with_missing_timezone(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'meter' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    with pytest.raises(ValueError):
        cls = BillingBaselineData(df, is_electricity_data=True)

# Check that a missing datetime index and column raises a Value Error
def test_billing_baseline_data_with_missing_datetime_index_and_column():

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(365)
    meter_value = np.random.rand(365)

    # Create the DataFrame
    df = pd.DataFrame(data={'meter' : meter_value, 'temperature': temperature_mean})

    with pytest.raises(ValueError):
        cls = BillingBaselineData(df, is_electricity_data=True)

@pytest.mark.parametrize('get_datetime_index', [['H', True]], indirect=True)
def test_billing_baseline_data_with_same_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0

@pytest.mark.parametrize('get_datetime_index', [['30T', True]], indirect=True)
def test_billing_baseline_data_with_same_half_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = BillingBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0

@pytest.mark.parametrize('get_datetime_index', [['30T', True],['H', True], ['D', True]], indirect=True)
def test_billing_reporting_data_with_missing_half_hourly_hourly_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

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
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 3