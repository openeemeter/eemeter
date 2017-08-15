import tempfile

from numpy.testing import assert_allclose
import pandas as pd
import pytest

from eemeter.weather import CZ2010WeatherSource
from eemeter.testing import MockWeatherClient


@pytest.fixture
def mock_cz2010_weather_source():
    tmp_url = "sqlite:///{}/weather_cache.db".format(tempfile.mkdtemp())
    ws = CZ2010WeatherSource("724838", tmp_url, preload=False)
    ws.client = MockWeatherClient()
    ws._load_data()
    return ws


def test_hourly_by_index(mock_cz2010_weather_source):
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='H')
    temps = mock_cz2010_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [35.617314, 35.607637])


def test_daily_by_index(mock_cz2010_weather_source):
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='D')
    temps = mock_cz2010_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [35.507046, 35.281477])

    # force load from cache
    mock_cz2010_weather_source._load_data()


def test_weird_frequency_by_index(mock_cz2010_weather_source):
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='5H')
    with pytest.raises(ValueError):
        mock_cz2010_weather_source.indexed_temperatures(index, 'degF')


def test_cross_year_boundary(mock_cz2010_weather_source):
    index = pd.date_range('1999-12-31 12:00:00Z', periods=2, freq='D')
    temps = mock_cz2010_weather_source.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [35.739576, 35.507046])


def test_bad_station():
    with pytest.raises(ValueError):
        CZ2010WeatherSource("INVALID")


def test_repr(mock_cz2010_weather_source):
    assert 'CZ2010WeatherSource("724838")' == str(mock_cz2010_weather_source)


def test_real_load():
    ws = CZ2010WeatherSource("725845")
    assert ws.tempC.shape == (8760,)
    import pdb; pdb.set_trace()
    assert ws.tempC.notnull().sum() == 8760
