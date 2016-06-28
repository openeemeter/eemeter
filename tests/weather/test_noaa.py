import tempfile

from numpy.testing import assert_allclose
import pandas as pd
import pytest

from eemeter.weather import GSODWeatherSource, ISDWeatherSource
from eemeter.testing import MockWeatherClient


@pytest.fixture
def mock_gsod_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = GSODWeatherSource("722880", tmp_dir)
    ws.client = MockWeatherClient()
    return ws


@pytest.fixture
def mock_isd_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = ISDWeatherSource("722880", tmp_dir)
    ws.client = MockWeatherClient()
    return ws


def test_gsod_index_hourly(mock_gsod_weather_source):
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='H')
    with pytest.raises(ValueError):
        mock_gsod_weather_source.indexed_temperatures(index, 'degF')


def test_gsod_index_daily(mock_gsod_weather_source):
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='D')
    temps = mock_gsod_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [32, 32])


def test_bad_gsod_station():
    with pytest.raises(ValueError):
        GSODWeatherSource("INVALID")


def test_isd_index_hourly(mock_isd_weather_source):
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='H')
    temps = mock_isd_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [32, 32])


def test_isd_index_daily(mock_isd_weather_source):
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='D')
    temps = mock_isd_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [32, 32])


def test_bad_isd_station():
    with pytest.raises(ValueError):
        ISDWeatherSource("INVALID")


def test_gsod_repr(mock_gsod_weather_source):
    assert str(mock_gsod_weather_source) == 'GSODWeatherSource("722880")'


def test_isd_repr(mock_isd_weather_source):
    assert str(mock_isd_weather_source) == 'ISDWeatherSource("722880")'
