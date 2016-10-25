import tempfile

from numpy.testing import assert_allclose
import pandas as pd
import pytest

from eemeter.weather import TMY3WeatherSource
from eemeter.testing import MockWeatherClient


@pytest.fixture
def mock_tmy3_weather_source():
    tmp_dir = tempfile.mkdtemp()
    ws = TMY3WeatherSource("724838", tmp_dir, preload=False)
    ws.client = MockWeatherClient()
    ws._load_data()
    return ws


def test_hourly_by_index(mock_tmy3_weather_source):
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='H')
    temps = mock_tmy3_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [32, 32])


def test_daily_by_index(mock_tmy3_weather_source):
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='D')
    temps = mock_tmy3_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [32, 32])

    # force load from cache
    mock_tmy3_weather_source._load_data()


def test_weird_frequency_by_index(mock_tmy3_weather_source):
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='5H')
    with pytest.raises(ValueError):
        mock_tmy3_weather_source.indexed_temperatures(index, 'degF')


def test_cross_year_boundary(mock_tmy3_weather_source):
    index = pd.date_range('1999-12-31 12:00:00Z', periods=2, freq='D')
    temps = mock_tmy3_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [32, 32])


def test_bad_station():
    with pytest.raises(ValueError):
        TMY3WeatherSource("INVALID")


def test_repr(mock_tmy3_weather_source):
    assert 'TMY3WeatherSource("724838")' == str(mock_tmy3_weather_source)
