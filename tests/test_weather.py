from eemeter.weather import WeatherSourceBase
from eemeter.weather import GSODWeatherSource
from eemeter.weather import WeatherUndergroundWeatherSource
from eemeter.weather import nrel_tmy3_station_from_lat_long
from eemeter.weather import ziplocate_us

from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import Consumption
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas

from datetime import datetime
import pytest
import os
import warnings

EPSILON = 10e-6

@pytest.fixture
def consumption_history_one_summer_electricity():
    c_list = [Consumption(1600,"kWh",electricity,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(1700,"kWh",electricity,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(1800,"kWh",electricity,datetime(2012,8,1),datetime(2012,9,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture(params=[(41.8955360374983,-87.6217660821178,"725340"),
                        (34.1678563835543,-118.126220490392,"722880"),
                        (42.3769095103979,-71.1247640734676,"725090")])
def lat_long_station(request):
    return request.param

@pytest.fixture(params=[(41.8955360374983,-87.6217660821178,"60611"),
                        (34.1678563835543,-118.126220490392,"91104"),
                        (42.3769095103979,-71.1247640734676,"02138"),
                        (None,None,"00000")])
def lat_long_zipcode(request):
    return request.param


def test_weather_source_base(consumption_history_one_summer_electricity):
    weather_source = WeatherSourceBase()
    with pytest.raises(NotImplementedError):
        hdd = weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity)

@pytest.mark.slow
def test_gsod_weather_source(consumption_history_one_summer_electricity):
    gsod_weather_source = GSODWeatherSource('722874-93134',start_year=2012,end_year=2012)
    avg_temps = gsod_weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity)
    assert abs(avg_temps[0] - 66.3833333333) < EPSILON
    assert abs(avg_temps[1] - 67.8032258065) < EPSILON
    assert abs(avg_temps[2] - 74.4451612903) < EPSILON

@pytest.mark.slow
def test_weather_underground_weather_source(consumption_history_one_summer_electricity):
    wunderground_api_key = os.environ.get('WEATHERUNDERGROUND_API_KEY')
    if wunderground_api_key:
        wu_weather_source = WeatherUndergroundWeatherSource('60605',
                                                            datetime(2012,6,1),
                                                            datetime(2012,10,1),
                                                            wunderground_api_key)
        avg_temps = wu_weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity)
        assert abs(avg_temps[0] - 74.4333333333) < EPSILON
        assert abs(avg_temps[1] - 82.6774193548) < EPSILON
        assert abs(avg_temps[2] - 75.4516129032) < EPSILON
    else:
        warnings.warn("Skipping WeatherUndergroundWeatherSource tests. "
            "Please set the environment variable "
            "WEATHERUNDERGOUND_API_KEY to run the tests.")

@pytest.mark.slow
def test_nrel_tmy3_station_from_lat_long(lat_long_station):
    lat,lng,station = lat_long_station
    nrel_api_key = os.environ.get('NREL_API_KEY')
    if nrel_api_key:
        assert station == nrel_tmy3_station_from_lat_long(lat,lng,nrel_api_key)
    else:
        warnings.warn("Skipping NREL tests. "
                "Please set the environment variable "
                "NREL_API_KEY to run the tests.")

@pytest.mark.slow
def test_ziplocate_us(lat_long_zipcode):
    lat,lng,zipcode = lat_long_zipcode
    if not lat or not lng:
        with pytest.raises(ValueError):
            ziplocate_us(zipcode)
    else:
        zip_lat, zip_lng = ziplocate_us(zipcode)
        assert abs(lat - zip_lat) < EPSILON
        assert abs(lng - zip_lng) < EPSILON
