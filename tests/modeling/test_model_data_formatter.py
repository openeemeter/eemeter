import tempfile
from datetime import datetime

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytz

from eemeter.weather import ISDWeatherSource
from eemeter.testing.mocks import MockWeatherClient
from eemeter.modeling.formatters import ModelDataFormatter
from eemeter.structures import EnergyTrace


@pytest.fixture
def mock_isd_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = ISDWeatherSource("722880", tmp_dir)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def daily_trace():
    data = {"value": [1, 1, np.nan], "estimated": [False, False, False]}
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=3, freq='D', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def hourly_trace():
    data = {"value": [1, 1, np.nan], "estimated": [False, False, False]}
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=3, freq='H', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


def test_basic_daily(daily_trace, mock_isd_weather_source):
    mdf = ModelDataFormatter("D")

    df = mdf.create_input(daily_trace, mock_isd_weather_source)

    assert all(df.columns == ["energy", "tempF"])
    assert df.index[0] == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert df.index[2] == datetime(2000, 1, 3, tzinfo=pytz.UTC)
    assert df.index.freq == 'D'
    assert_allclose(df.energy, [1, 1, np.nan])
    assert_allclose(df.tempF, [32., 32., 32.])


def test_basic_hourly(hourly_trace, mock_isd_weather_source):
    mdf = ModelDataFormatter("H")

    df = mdf.create_input(hourly_trace, mock_isd_weather_source)

    assert all(df.columns == ["energy", "tempF"])
    assert df.index[0] == datetime(2000, 1, 1, 0, tzinfo=pytz.UTC)
    assert df.index[2] == datetime(2000, 1, 1, 2, tzinfo=pytz.UTC)
    assert df.index.freq == 'H'
    assert_allclose(df.energy, [1, 1, np.nan])
    assert_allclose(df.tempF, [32., 32., 32.])


def test_basic_hourly_to_daily(hourly_trace, mock_isd_weather_source):
    mdf = ModelDataFormatter("D")

    df = mdf.create_input(hourly_trace, mock_isd_weather_source)

    assert all(df.columns == ["energy", "tempF"])
    assert df.index[0] == datetime(2000, 1, 1, 0, tzinfo=pytz.UTC)
    assert df.index.freq == 'D'
    assert_allclose(df.energy, [2])
    assert_allclose(df.tempF, [32.])


def test_daily_to_hourly_fails(daily_trace, mock_isd_weather_source):
    mdf = ModelDataFormatter("H")

    with pytest.raises(ValueError):
        mdf.create_input(daily_trace, mock_isd_weather_source)

def test_daily_demand_fixture(daily_trace, mock_isd_weather_source):
    mdf = ModelDataFormatter("D")

    df = mdf.create_demand_fixture(daily_trace.data.index,
                                   mock_isd_weather_source)

    assert all(df.columns == ["tempF"])
    assert df.index[0] == datetime(2000, 1, 1, 0, tzinfo=pytz.UTC)
    assert df.index[2] == datetime(2000, 1, 3, tzinfo=pytz.UTC)
    assert df.index.freq == 'D'
    assert_allclose(df.tempF, [32., 32., 32.])

def test_hourly_demand_fixture(hourly_trace, mock_isd_weather_source):
    mdf = ModelDataFormatter("H")

    df = mdf.create_demand_fixture(hourly_trace.data.index,
                                   mock_isd_weather_source)

    assert all(df.columns == ["tempF"])
    assert df.index[0] == datetime(2000, 1, 1, 0, tzinfo=pytz.UTC)
    assert df.index[2] == datetime(2000, 1, 1, 2, tzinfo=pytz.UTC)
    assert df.index.freq == 'H'
    assert_allclose(df.tempF, [32., 32., 32.])
