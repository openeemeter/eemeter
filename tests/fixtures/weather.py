import pytest

from eemeter.weather import GSODWeatherSource
from eemeter.weather import TMY3WeatherSource

from pkg_resources import resource_stream
import os

@pytest.fixture(scope="session")
def gsod_722880_2012_2014_weather_source():

    ws = GSODWeatherSource("722880")
    ws.add_year_range(2012, 2014)
    return ws

@pytest.fixture(scope="session")
def tmy3_722880_weather_source():
    return TMY3WeatherSource("722880")
