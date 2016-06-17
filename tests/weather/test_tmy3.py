from eemeter.weather import TMY3WeatherSource
import pandas as pd
from numpy.testing import assert_allclose
import tempfile
import pytest


def test_hourly_by_index():
    ws = TMY3WeatherSource("724838")
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='H')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [46.4, 46.4])


def test_daily_by_index():
    ws = TMY3WeatherSource("724838")
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='D')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [46.175, 45.5])


def test_cross_year_boundary():
    ws = TMY3WeatherSource("724838")
    index = pd.date_range('1999-12-31 12:00:00Z', periods=2, freq='D')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [47.15, 46.175])


def test_bad_isd_station():
    with pytest.raises(ValueError):
        TMY3WeatherSource("INVALID")


def test_cache():
    tmp_dir = tempfile.mkdtemp()
    ws = TMY3WeatherSource("724838", tmp_dir)
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='D')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [46.175, 45.5])
