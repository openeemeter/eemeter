import tempfile
from datetime import datetime

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytz

from eemeter.weather import ISDWeatherSource
from eemeter.testing.mocks import MockWeatherClient
from eemeter.modeling.formatters import CaltrackFormatter
from eemeter.structures import EnergyTrace


@pytest.fixture
def mock_isd_weather_source():
    tmp_url = "sqlite:///{}/weather_cache.db".format(tempfile.mkdtemp())
    ws = ISDWeatherSource("722880", tmp_url)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def daily_trace():
    data = {"value": np.ones(60) * 1}
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=60, freq='D', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def hourly_trace():
    data = {"value": np.ones(60 * 24) * 1}
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=60 * 24, freq='H', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def daily_trace_with_nans():
    data = {"value": np.append(np.ones(30) * 1, np.ones(30) * np.nan)}
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=60, freq='D', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


def test_basic_daily(daily_trace, mock_isd_weather_source):
    mdf = CaltrackFormatter()

    df = mdf.create_input(daily_trace, mock_isd_weather_source)

    assert 'upd' in df.columns
    assert 'CDD_70' in df.columns
    assert 'HDD_60' in df.columns
    assert df.index[0] == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert df.index[1] == datetime(2000, 2, 1, tzinfo=pytz.UTC)
    assert len(df.index) == 2
    assert_allclose(df.upd, [1, 1])
    assert_allclose(df.HDD_60, [60 - 32., 60 - 32.])

    description = mdf.describe_input(df)
    assert description.get('start_date') == \
        datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert description.get('n_rows') == 2


def test_basic_hourly(hourly_trace, mock_isd_weather_source):
    mdf = CaltrackFormatter()

    df = mdf.create_input(hourly_trace, mock_isd_weather_source)

    assert 'upd' in df.columns
    assert 'CDD_70' in df.columns
    assert 'HDD_60' in df.columns
    assert df.index[0] == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert df.index[1] == datetime(2000, 2, 1, tzinfo=pytz.UTC)
    assert len(df.index) == 2
    assert_allclose(df.upd, [24, 24])
    assert_allclose(df.HDD_60, [60 - 32., 60 - 32.])

    description = mdf.describe_input(df)
    assert description.get('start_date') == \
        datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert description.get('n_rows') == 2


def test_basic_daily_varbp(daily_trace, mock_isd_weather_source):
    mdf = CaltrackFormatter(grid_search=True)

    df = mdf.create_input(daily_trace, mock_isd_weather_source)

    assert 'upd' in df.columns
    assert 'CDD_50' in df.columns
    assert 'CDD_55' in df.columns
    assert 'CDD_60' in df.columns
    assert 'CDD_65' in df.columns
    assert 'CDD_70' in df.columns
    assert 'CDD_75' in df.columns
    assert 'CDD_80' in df.columns
    assert 'CDD_85' in df.columns
    assert 'HDD_50' in df.columns
    assert 'HDD_55' in df.columns
    assert 'HDD_60' in df.columns
    assert 'HDD_65' in df.columns
    assert 'HDD_70' in df.columns
    assert 'HDD_75' in df.columns
    assert 'HDD_80' in df.columns
    assert 'HDD_85' in df.columns
    assert df.index[0] == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert df.index[1] == datetime(2000, 2, 1, tzinfo=pytz.UTC)
    assert len(df.index) == 2
    assert_allclose(df.upd, [1, 1])
    assert_allclose(df.HDD_60, [60 - 32., 60 - 32.])

    description = mdf.describe_input(df)
    assert description.get('start_date') == \
        datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert description.get('n_rows') == 2


def test_basic_daily_with_nans(daily_trace_with_nans, mock_isd_weather_source):
    mdf = CaltrackFormatter()

    df = mdf.create_input(daily_trace_with_nans, mock_isd_weather_source)

    assert 'upd' in df.columns
    assert 'CDD_70' in df.columns
    assert 'HDD_60' in df.columns
    assert df.index[0] == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert df.index[1] == datetime(2000, 2, 1, tzinfo=pytz.UTC)
    assert len(df.index) == 2
    assert_allclose(df.upd, [1, np.nan])
    assert_allclose(df.HDD_60, [60 - 32., np.nan])

    description = mdf.describe_input(df)
    assert description.get('start_date') == \
        datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert description.get('n_rows') == 2


def test_daily_demand_fixture(daily_trace, mock_isd_weather_source):
    mdf = CaltrackFormatter()

    df = mdf.create_demand_fixture(daily_trace.data.index,
                                   mock_isd_weather_source)

    assert 'CDD_70' in df.columns
    assert 'HDD_60' in df.columns
    assert df.index[0] == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert df.index[1] == datetime(2000, 2, 1, tzinfo=pytz.UTC)
    assert_allclose(df.HDD_60, [60 - 32., 60 - 32.])

    description = mdf.describe_input(df)
    assert description.get('start_date') == \
        datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert description.get('n_rows') == 2


def test_repr():
    mdf = CaltrackFormatter()
    assert str(mdf) == 'CaltrackFormatter()'


def test_empty_description():
    mdf = CaltrackFormatter()
    description = mdf.describe_input(pd.DataFrame())
    assert description['start_date'] is None
    assert description['end_date'] is None
    assert description['n_rows'] is 0
