from eemeter.weather import ISDWeatherSource
from eemeter.testing.mocks import MockWeatherClient
from eemeter.modeling.formatters import ModelDataFormatter

from eemeter.modeling.models import HourlyDayOfWeekModel
import numpy as np
import pytest
import pandas as pd
import pytz
import tempfile
import eemeter.modeling.exceptions as model_exceptions

@pytest.fixture
def mock_isd_weather_source():
    tmp_url = "sqlite:///{}/weather_cache.db".format(tempfile.mkdtemp())
    ws = ISDWeatherSource("722880", tmp_url)
    ws.client = MockWeatherClient()
    return ws

@pytest.fixture
def hourly_trace_with_dummy_energy():
    date_hr_timestamp = pd.date_range('2017-09-16', periods=72, freq='H', tz=pytz.UTC)
    df = pd.DataFrame( {'energy' : [1.0 for xx in date_hr_timestamp]} , index=date_hr_timestamp)
    return df

@pytest.fixture
def input_df(mock_isd_weather_source, hourly_trace_with_dummy_energy):
    tempF = mock_isd_weather_source.indexed_temperatures(hourly_trace_with_dummy_energy.index, "degF")
    return hourly_trace_with_dummy_energy.assign(tempF=tempF)

def test_add_time_day():
    # Creating hourly time stamp for three days
    # 2017-09-16 ==> Saturday
    # 2017-09-17 ==> Sunday
    # 2017-09-18 ==> Monday
    date_hr_timestamp = pd.date_range('2017-09-16', periods=72, freq='H', tz=pytz.UTC)
    df = pd.DataFrame( {'energy' : [1.0 for xx in date_hr_timestamp]} , index=date_hr_timestamp)
    day_of_week = HourlyDayOfWeekModel()
    returned_df = day_of_week.add_time_day(df)
    assert 'hour_of_day' in returned_df
    assert 'day_of_week' in returned_df

    # Testing day of week columns
    # 2017-09-16 is Saturday and so day of week value of the first row
    # in returned_df should be 5
    assert returned_df.at[returned_df.index[0], 'day_of_week'] == '5'
    #2017-09-19 is Monday and so day of week value of last row should 0
    assert returned_df.at[returned_df.index[-1], 'day_of_week'] == '0'
    # 2017-09-18 is Sunday, day_of_week should be 6
    assert returned_df.at[returned_df.index[25], 'day_of_week'] == '6'

    # First hour of 2017-09-16
    assert returned_df.at[returned_df.index[1], 'hour_of_day'] == '1'
    # Second hour of 2017-09-16
    assert returned_df.at[returned_df.index[2], 'hour_of_day'] == '2'


def test_add_hdd(input_df):
    hdd_function = HourlyDayOfWeekModel()
    hdd_val= hdd_function.add_hdd(input_df)
    assert 'hdd' in hdd_val

def test_add_cdd(input_df):
    cdd_function = HourlyDayOfWeekModel()
    cdd_val= cdd_function.add_cdd(input_df)
    assert 'cdd' in cdd_val

def test_predict(input_df):
    model = HourlyDayOfWeekModel(min_contiguous_months=0)
    model.fit(input_df)

    # Test cases on the output data types of predict function.
    # When summed = True, prediction and variance are atomic
    # floating point numbers
    prediction, variance= model.predict(input_df, summed=True)
    assert type(prediction) == np.float64
    #assert type(variance) == np.float64

    # When summed=False, prediction and variance are pandas Series.
    prediction, variance= model.predict(input_df, summed=False)
    assert type(prediction) == pd.Series
    assert type(variance) == pd.Series
    # We expect out linear regression model to make prediction between 0.95 & 2.0 because the energy
    # consumed in input_df dataframe is always 1.0, please take a look at hourly_trace_with_dummy_energy function
    assert prediction[0].item() > 0.95 and prediction[0].item()  < 2.0

def test_min_contiguous_months(input_df):
    min_contiguous_months = 9
    model = HourlyDayOfWeekModel(min_contiguous_months=min_contiguous_months)
    with pytest.raises(model_exceptions.DataSufficiencyException) as sufficiency_exception:
        model.fit(input_df)

