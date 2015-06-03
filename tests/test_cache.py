import os

from eemeter.weather import GSODWeatherSource
from eemeter.weather import ISDWeatherSource
from eemeter.weather import TMY3WeatherSource

import pytest

os.environ["EEMETER_WEATHER_CACHE_DATABASE_URL"] = 'sqlite:///:memory:'

ws_pk = None

@pytest.mark.slow
def test_gsod_weather_cache():
    ws = GSODWeatherSource("722660",2012,2012)

    assert 366 == len(ws.data)
    assert 366 == len(ws.get_temperature_set().fetchall())

    global ws_pk
    ws_pk = ws.weather_station_pk

@pytest.mark.slow
def test_gsod_weather_cache_wide_date_range():
    ws = GSODWeatherSource("722660",2003,2012)

    assert 3653 == len(ws.data)
    assert 4019 == len(ws.get_temperature_set().fetchall())
    global ws_pk
    assert ws.weather_station_pk == ws_pk

@pytest.mark.slow
def test_isd_weather_cache_00():

    global ws_pk

    ws = ISDWeatherSource("722660",2012,2012)
    assert 8783 == len(ws.data)
    assert 11652 == len(ws.get_temperature_set().fetchall())
    assert ws.weather_station_pk == ws_pk

    ws = ISDWeatherSource("722660",2013,2013)
    assert 17542 == len(ws.data)
    assert 23551 == len(ws.get_temperature_set().fetchall())
    assert ws.weather_station_pk == ws_pk

    # should be fast now
    for i in range(2):
        ws = ISDWeatherSource("722660",2012,2013)
        assert 17542 == len(ws.data)
        assert 23551 == len(ws.get_temperature_set().fetchall())
        assert ws.weather_station_pk == ws_pk

@pytest.mark.slow
def test_isd_weather_cache_01():
    ws = ISDWeatherSource("722660",2012,2012)
    assert 17542 == len(ws.data)
    assert 23551 == len(ws.get_temperature_set().fetchall())
    global ws_pk
    assert ws.weather_station_pk == ws_pk

@pytest.mark.slow
def test_tmy3_weather_cache():
    ws = TMY3WeatherSource("722660")
    assert 8759 == len(ws.data)
    assert 8759 == len(ws.get_temperature_set().fetchall())
    global ws_pk
    assert ws.weather_station_pk == ws_pk
