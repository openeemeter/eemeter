from eemeter.weather import GSODWeatherSource, ISDWeatherSource
import pandas as pd
import pytest
from numpy.testing import assert_allclose


def test_gsod_index_hourly():
    ws = GSODWeatherSource("722890")
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='H')
    with pytest.raises(ValueError):
        ws.indexed_temperatures(index, 'degF')

def test_gsod_index_daily():
    ws = GSODWeatherSource("722880")
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='D')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [45.7, 46.9])

def test_isd_index_hourly():
    ws = ISDWeatherSource("722880")
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='H')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [50., 50.])

def test_isd_index_daily():
    ws = ISDWeatherSource("722880")
    index = pd.date_range('2011-01-01 00:00:00Z', periods=2, freq='D')
    temps = ws.indexed_temperatures(index, 'degF')
    assert all(temps.index == index)
    assert all(temps.index == index)
    assert temps.shape == (2,)
    assert_allclose(temps.values, [45.7175, 46.88])
