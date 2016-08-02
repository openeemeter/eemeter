import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import pytz

from eemeter.testing.mocks import MockWeatherClient
from eemeter.weather import ISDWeatherSource
from eemeter.modeling.formatters import ModelDataBillingFormatter
from eemeter.structures import EnergyTrace


@pytest.fixture
def mock_isd_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = ISDWeatherSource("722880", tmp_dir)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def trace1():
    data = {
        "value": [1, 1, 1, 1, np.nan],
        "estimated": [False, False, True, False, False]
    }
    columns = ["value", "estimated"]
    index = [
        datetime(2011, 1, 1, tzinfo=pytz.UTC),
        datetime(2011, 2, 1, tzinfo=pytz.UTC),
        datetime(2011, 3, 2, tzinfo=pytz.UTC),
        datetime(2011, 4, 3, tzinfo=pytz.UTC),
        datetime(2011, 4, 29, tzinfo=pytz.UTC),
    ]
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def trace2():
    data = {
        "value": [np.nan],
        "estimated": [True]
    }
    columns = ["value", "estimated"]
    index = [
        datetime(2011, 1, 1, tzinfo=pytz.UTC),
    ]
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def trace3():
    data = {
        "value": [1, np.nan],
        "estimated": [True, False]
    }
    columns = ["value", "estimated"]
    index = [
        datetime(2011, 1, 1, tzinfo=pytz.UTC),
        datetime(2011, 2, 1, tzinfo=pytz.UTC),
    ]
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


def test_basic_daily(trace1, mock_isd_weather_source):
    mdbf = ModelDataBillingFormatter()
    trace_data, temperature_data = mdbf.create_input(
        trace1, mock_isd_weather_source)
    assert trace_data.shape == (4,)
    assert temperature_data.shape == (2832, 1)


def test_empty(trace2, mock_isd_weather_source):
    mdbf = ModelDataBillingFormatter()
    trace_data, temperature_data = mdbf.create_input(
        trace2, mock_isd_weather_source)
    assert trace_data.shape == (0,)
    assert temperature_data.shape == (0,)


def test_small(trace3, mock_isd_weather_source):
    mdbf = ModelDataBillingFormatter()
    with pytest.raises(ValueError):
        mdbf.create_input(trace3, mock_isd_weather_source)
