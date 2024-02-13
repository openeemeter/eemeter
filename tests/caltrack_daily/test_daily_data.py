from eemeter.eemeter.models.daily.data import DailyBaselineData

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

# @pytest.mark.parametrize('get_datetime_index', [['D', False]], indirect=True)
# def test_daily_baseline_data_with_same_daily_frequencies(get_datetime_index):
#     datetime_index = get_datetime_index

#     # Create a 'temperature_mean' and meter_value columns with random data
#     temperature_mean = np.random.rand(len(datetime_index))
#     meter_value = np.random.rand(len(datetime_index))

#     # Create the DataFrame
#     df = pd.DataFrame(data={'meter_value' : meter_value, 'temperature_mean': temperature_mean}, index=datetime_index)

#     cls = DailyBaselineData(df, is_electricity_data=True)

#     assert cls._baseline_meter_df is not None
#     assert len(cls.warnings) == 1
#     assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
#     assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_datetime_column():
    pass


@pytest.mark.parametrize('get_datetime_index', [['D', True]], indirect=True)
def test_daily_baseline_data_with_same_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls._baseline_meter_df is not None
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    assert len(cls.disqualification) == 0

@pytest.mark.parametrize('get_datetime_index', [['H', True]], indirect=True)
def test_daily_baseline_data_with_same_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls._baseline_meter_df is not None
    assert len(cls.warnings) == 1
    # TODO : this seems like a pre-existing bug in the 'as_freq' method, why is the last element set as null in the high frequency data?
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.missing_high_frequency_meter_data'
    assert len(cls.disqualification) == 0

@pytest.mark.parametrize('get_datetime_index', [['30T', True]], indirect=True)
def test_daily_baseline_data_with_same_half_hourly_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls._baseline_meter_df is not None
    assert len(cls.warnings) == 1
    # TODO : this seems like a pre-existing bug in the 'as_freq' method, why is the last element set as null in the high frequency data?
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.missing_high_frequency_meter_data'
    assert len(cls.disqualification) == 0

def test_daily_baseline_data_with_daily_and_half_hourly_frequencies(get_temperature_data_half_hourly, get_meter_data_daily):
    # Create a DataFrame with uneven frequency
    df = get_temperature_data_half_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls._baseline_meter_df is not None
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_daily_and_hourly_frequencies(get_meter_data_daily, get_temperature_data_hourly):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls._baseline_meter_df is not None
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0
