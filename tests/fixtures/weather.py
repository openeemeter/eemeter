import pytest

from eemeter.weather import GSODWeatherSource
from eemeter.weather import TMY3WeatherSource

from pkg_resources import resource_stream
import os

@pytest.fixture(scope="session")
def gsod_722880_2012_2014_weather_source():

    filenames = [
            '722880-23152-2012.op.gz',
            '722880-23152-2013.op.gz',
            '722880-23152-2014.op.gz']
    gz_filenames = []
    for fn in filenames:
        with resource_stream('eemeter.resources', fn) as gzf:
            gz_filenames.append(gzf.name)
    return GSODWeatherSource(station_id="722880", gz_filenames=gz_filenames)

    #    return GSODWeatherSource('722880',start_year=2012,end_year=2014)

@pytest.fixture(scope="session")
def tmy3_722880_weather_source():
    return TMY3WeatherSource('722880')
