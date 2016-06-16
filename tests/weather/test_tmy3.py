from eemeter.weather import TMY3WeatherSource
import pandas as pd
from numpy.testing import assert_allclose
import pytest

def test_hourly_by_index():
    ws = TMY3WeatherSource("722890")
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='H')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [33.08, 32.00])

def test_daily_by_index():
    ws = TMY3WeatherSource("722890")
    index = pd.date_range('2000-01-01 00:00:00Z', periods=2, freq='D')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [31.7525, 30.1175])

def test_cross_year_boundary():
    ws = TMY3WeatherSource("722890")
    index = pd.date_range('1999-12-31 12:00:00Z', periods=2, freq='D')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [32.5925, 31.7525])
