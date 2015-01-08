import pytest

from eemeter.weather import GSODWeatherSource
from eemeter.weather import TMY3WeatherSource

@pytest.fixture
def gsod_722880_2012_2014_weather_source():
    return GSODWeatherSource('722880',start_year=2012,end_year=2014)

@pytest.fixture
def tmy3_722880_weather_source():
    return TMY3WeatherSource('722880',os.environ.get("TMY3_DIRECTORY"))
