from eemeter.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from eemeter import load_sample
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def get_datetime_index(request):
    # Request = [frequency , is_timezone_aware]

    # Create a DateTimeIndex at 30-minute intervals
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
def get_datetime_index_daily_without_timezone():
    # Create a DateTimeIndex at daily intervals
    datetime_index = pd.date_range(start='2023-01-01', end='2024-01-01', inclusive="left", freq='D')

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

@pytest.fixture
def get_temperature_data_daily(get_datetime_index_daily_with_timezone):
    datetime_index = get_datetime_index_daily_with_timezone

    # Create a 'temperature_mean' column with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    return df

# Check that a missing timezone raises a Value Error
@pytest.mark.parametrize('get_datetime_index', [['D', False]], indirect=True)
def test_daily_baseline_data_with_missing_timezone(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'meter' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    with pytest.raises(ValueError):
        cls = DailyBaselineData(df, is_electricity_data=True)

# Check that a missing datetime index and column raises a Value Error
def test_daily_baseline_data_with_missing_datetime_index_and_column():

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(365)
    meter_value = np.random.rand(365)

    # Create the DataFrame
    df = pd.DataFrame(data={'meter' : meter_value, 'temperature': temperature_mean})

    with pytest.raises(ValueError):
        cls = DailyBaselineData(df, is_electricity_data=True)

@pytest.mark.parametrize('get_datetime_index', [['D', True]], indirect=True)
def test_daily_baseline_data_with_datetime_column(get_datetime_index):
    df = pd.DataFrame()
    df['datetime'] = get_datetime_index
    df['temperature'] = np.random.rand(len(get_datetime_index))
    df['observed'] = np.random.rand(len(get_datetime_index))

    cls = DailyBaselineData(df, is_electricity_data=True)
    
    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    assert len(cls.disqualification) == 0

@pytest.mark.parametrize('get_datetime_index', [['D', True]], indirect=True)
def test_daily_baseline_data_with_same_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))
    meter_value = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'observed' : meter_value, 'temperature': temperature_mean}, index=datetime_index)

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
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

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 0
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

    assert cls.df is not None
    assert len(cls.df) == 365
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

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


def test_daily_baseline_data_with_daily_and_hourly_frequencies(get_meter_data_daily, get_temperature_data_hourly):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0

def test_daily_baseline_data_with_specific_hourly_input():
    meter, temperature, _ = load_sample('il-electricity-cdd-hdd-hourly')
    meter = meter[meter.index.year==2017]
    temperature = temperature[temperature.index.year==2017]
    cls = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == ['eemeter.data_quality.utc_index', 'eemeter.caltrack_sufficiency_criteria.extreme_values_detected']
    assert len(cls.disqualification) == 0

def test_daily_baseline_data_with_specific_daily_input():
    meter, temperature, _ = load_sample('il-electricity-cdd-hdd-daily')
    meter = meter[meter.index.year==2017]
    temperature = temperature[temperature.index.year==2017]
    cls = DailyBaselineData.from_series(meter, temperature, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 2
    assert [warning.qualified_name for warning in cls.warnings] == ['eemeter.data_quality.utc_index', 'eemeter.caltrack_sufficiency_criteria.extreme_values_detected']
    assert len(cls.disqualification) == 0

def test_daily_baseline_data_with_missing_hourly_temperature_data(get_meter_data_daily, get_temperature_data_hourly):
    df = get_temperature_data_hourly

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])
    
    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    df.loc[df[mask].sample(frac=0.6).index, 'temperature'] = np.nan

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 0

    # TODO : BUG : the 'compute_temperature_features' method in features.py does not add a warning for missing high frequency data. Should be fixed.
    assert len(cls.disqualification) == 2
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)

def test_daily_baseline_data_with_missing_half_hourly_temperature_data(get_meter_data_daily, get_temperature_data_half_hourly):
    df = get_temperature_data_half_hourly

    # Create a mask for Tuesdays and Thursdays
    mask = df.index.dayofweek.isin([1, 3])
    
    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df[mask].sample(frac=0.6).index, 'temperature'] = np.nan

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.missing_high_frequency_temperature_data'
    assert len(cls.disqualification) == 3
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data', 'eemeter.caltrack_sufficiency_criteria.missing_monthly_temperature_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)

def test_daily_baseline_data_with_missing_daily_temperature_data(get_meter_data_daily, get_temperature_data_daily):
    df = get_temperature_data_daily
    
    # Set 60% of the temperature data as missing on Tuesdays and Thursdays
    # This should cause the high frequency temperature check to fail on these days
    df.loc[df.index.dayofweek.isin([1,3]), 'temperature'] = np.nan

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    assert len(cls.disqualification) == 3
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data', 'eemeter.caltrack_sufficiency_criteria.missing_monthly_temperature_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)

def test_daily_baseline_data_with_missing_meter_data(get_meter_data_daily, get_temperature_data_hourly):
    df = get_temperature_data_hourly

    # Create a DataFrame with daily frequency
    df_meter = get_meter_data_daily

    # Set Tuesdays & Thursdays data as missing
    df_meter.loc[df_meter.index.dayofweek.isin([1,3]), 'observed'] = np.nan

    # Merge 'df' and 'df_meter' in an outer join
    df = df.merge(df_meter, left_index=True, right_index=True, how='outer')

    cls = DailyBaselineData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 0
    # assert all(warning.qualified_name in expected_warnings for warning in cls.warnings)
    assert len(cls.disqualification) == 3
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.missing_monthly_meter_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_meter_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)

@pytest.mark.parametrize('get_datetime_index', [['30T', True],['H', True]], indirect=True)
def test_daily_reporting_data_with_half_hourly_and_hourly_daily_frequencies(get_datetime_index):
    datetime_index = get_datetime_index

    # Create a 'temperature_mean' and meter_value columns with random data
    temperature_mean = np.random.rand(len(datetime_index))

    # Create the DataFrame
    df = pd.DataFrame(data={'temperature': temperature_mean}, index=datetime_index)

    cls = DailyReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 0
    assert len(cls.disqualification) == 0


@pytest.mark.parametrize('get_datetime_index', [['30T', True],['H', True]], indirect=True)
def test_daily_reporting_data_with_missing_half_hourly_and_hourly_frequencies(get_datetime_index):
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

    cls = DailyReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.missing_high_frequency_temperature_data'
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.missing_monthly_temperature_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)


@pytest.mark.parametrize('get_datetime_index', [['D', True]], indirect=True)
def test_daily_reporting_data_with_missing_daily_frequencies(get_datetime_index):
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

    cls = DailyReportingData(df, is_electricity_data=True)

    assert cls.df is not None
    assert len(cls.df) == 365
    assert len(cls.warnings) == 1
    assert cls.warnings[0].qualified_name == 'eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency'
    expected_disqualifications = ['eemeter.caltrack_sufficiency_criteria.missing_monthly_temperature_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_data', 'eemeter.caltrack_sufficiency_criteria.too_many_days_with_missing_temperature_data']
    assert all(disqualification.qualified_name in expected_disqualifications for disqualification in cls.disqualification)