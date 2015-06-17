import os
from datetime import datetime

from eemeter.weather import GSODWeatherSource
from eemeter.weather import ISDWeatherSource
from eemeter.weather import TMY3WeatherSource
from sqlalchemy import select

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

@pytest.mark.slow
def test_cache_deletes_old_records():
    ws = ISDWeatherSource("722660",2012,2012)

    # Make sure there are two records to begin with (this just happens to be
    # the case for this weather station at this particular hour - usually there
    # is only one record per hour.
    temperature_set = ws.get_temperature_set()
    assert 2 == sum([t.dt == datetime(2012,1,1,0) for t in temperature_set])

    # overwrite it
    records = [{"temp_C": 0, "dt": datetime(2012,1,1,0)}]
    ws.update_cache(records)

    # Now there should just be one
    temperature_set = ws.get_temperature_set()
    assert 1 == sum([t.dt == datetime(2012,1,1,0) for t in temperature_set])

