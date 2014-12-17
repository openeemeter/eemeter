from eemeter.weather import WeatherSourceBase
from eemeter.weather import GSODWeatherSource
from eemeter.weather import ISDWeatherSource
from eemeter.weather import TMY3WeatherSource
from eemeter.weather import WeatherUndergroundWeatherSource
from eemeter.weather import nrel_tmy3_station_from_lat_long
from eemeter.weather import ziplocate_us
from eemeter.weather import usaf_station_from_zipcode

from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import Consumption
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas

from datetime import datetime
import pytest
import os
import warnings

EPSILON = 10e-6

##### Fixtures #####

@pytest.fixture
def consumption_history_one_summer_electricity():
    c_list = [Consumption(1600,"kWh",electricity,datetime(2012,6,1),datetime(2012,7,1)),
              Consumption(1700,"kWh",electricity,datetime(2012,7,1),datetime(2012,8,1)),
              Consumption(1800,"kWh",electricity,datetime(2012,8,1),datetime(2012,9,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture(params=[(41.8955360374983,-87.6217660821178,"725340"),
                        (34.1678563835543,-118.126220490392,"722880"),
                        (42.3769095103979,-71.1247640734676,"725090"),
                        (42.3594006437094,-87.8581578622419,"725347")])
def lat_long_station(request):
    return request.param

@pytest.fixture(params=[(41.8955360374983,-87.6217660821178,"60611"),
                        (34.1678563835543,-118.126220490392,"91104"),
                        (42.3769095103979,-71.1247640734676,"02138"),
                        (42.3594006437094,-87.8581578622419,"60085"),
                        (None,None,"00000")])
def lat_long_zipcode(request):
    return request.param

@pytest.fixture(params=[('722874-93134',2012,2012),
                        ('722874',2012,2012)])
def gsod_weather_source(request):
    return request.param

@pytest.fixture(params=[('722874-93134',2012,2012),
                        ('722874',2012,2012)])
def isd_weather_source(request):
    return request.param

@pytest.fixture(params=[TMY3WeatherSource('722880',os.environ.get("TMY3_DIRECTORY"))])
def tmy3_weather_source(request):
    return request.param

@pytest.fixture(params=[("60611","725340"),
                        ("91104","722880"),
                        ("02138","725090"),
                        ("60085","725347")])
def zipcode_to_station(request):
    return request.param


##### Tests #####

def test_weather_source_base(consumption_history_one_summer_electricity):
    weather_source = WeatherSourceBase()
    with pytest.raises(NotImplementedError):
        avg_temps = weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity,"degF")
    with pytest.raises(NotImplementedError):
        hdds = weather_source.get_hdd(consumption_history_one_summer_electricity,electricity,"degF",base=65)

@pytest.mark.slow
def test_gsod_weather_source(consumption_history_one_summer_electricity,gsod_weather_source):
    gsod_weather_source = GSODWeatherSource(*gsod_weather_source)
    avg_temps = gsod_weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity,"degF")
    assert abs(avg_temps[0] - 66.3833333333) < EPSILON
    assert abs(avg_temps[1] - 67.8032258065) < EPSILON
    assert abs(avg_temps[2] - 74.4451612903) < EPSILON
    hdds = gsod_weather_source.get_hdd(consumption_history_one_summer_electricity,electricity,"degF",65)
    assert abs(hdds[0] - 6.3833333333) < EPSILON
    assert abs(hdds[1] - 7.8032258065) < EPSILON
    assert abs(hdds[2] - 14.4451612903) < EPSILON
    cdds = gsod_weather_source.get_cdd(consumption_history_one_summer_electricity,electricity,"degF",65)
    assert abs(cdds[0] - 0.0) < EPSILON
    assert abs(cdds[1] - 0.0) < EPSILON
    assert abs(cdds[2] - 0.0) < EPSILON

@pytest.mark.slow
def test_weather_underground_weather_source(consumption_history_one_summer_electricity):
    wunderground_api_key = os.environ.get('WEATHERUNDERGROUND_API_KEY')
    if wunderground_api_key:
        wu_weather_source = WeatherUndergroundWeatherSource('60605',
                                                            datetime(2012,6,1),
                                                            datetime(2012,10,1),
                                                            wunderground_api_key)
        avg_temps = wu_weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity,"degF")
        assert abs(avg_temps[0] - 74.4333333333) < EPSILON
        assert abs(avg_temps[1] - 82.6774193548) < EPSILON
        assert abs(avg_temps[2] - 75.4516129032) < EPSILON
        hdds = wu_weather_source.get_hdd(consumption_history_one_summer_electricity,electricity,"degF",65)
        assert abs(hdds[0] - 6.3833333333) < EPSILON
        assert abs(hdds[1] - 7.8032258065) < EPSILON
        assert abs(hdds[2] - 14.4451612903) < EPSILON
        cdds = wu_weather_source.get_cdd(consumption_history_one_summer_electricity,electricity,"degF",65)
        assert abs(cdds[0] - 0.0) < EPSILON
        assert abs(cdds[1] - 0.0) < EPSILON
        assert abs(cdds[2] - 0.0) < EPSILON
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

@pytest.mark.slow
def test_isd_weather_source(consumption_history_one_summer_electricity,isd_weather_source):
    isd_weather_source = ISDWeatherSource(*isd_weather_source)
    avg_temps = isd_weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity,"degF")
    assert abs(avg_temps[0] - 66.576956521739135) < EPSILON
    assert abs(avg_temps[1] - 68.047780898876411) < EPSILON
    assert abs(avg_temps[2] - 74.697162921348323) < EPSILON
    hdds = isd_weather_source.get_hdd(consumption_history_one_summer_electricity,electricity,"degF")
    assert abs(hdds[0] - 1.576956521739135) < EPSILON
    assert abs(hdds[1] - 3.047780898876411) < EPSILON
    assert abs(hdds[2] - 11.697162921348323) < EPSILON
    cdds = isd_weather_source.get_cdd(consumption_history_one_summer_electricity,electricity,"degF")
    assert abs(cdds[0] - 0.0) < EPSILON
    assert abs(cdds[1] - 0.0) < EPSILON
    assert abs(cdds[2] - 0.0) < EPSILON

@pytest.mark.slow
def test_usaf_station_from_zipcode(zipcode_to_station):
    zipcode,station = zipcode_to_station
    nrel_api_key = os.environ.get('NREL_API_KEY')
    if nrel_api_key:
        assert usaf_station_from_zipcode(zipcode,nrel_api_key) == station
    else:
        warnings.warn("Skipping NREL tests. "
                "Please set the environment variable "
                "NREL_API_KEY to run the tests.")

@pytest.mark.slow
def test_tmy3_weather_source(consumption_history_one_summer_electricity,tmy3_weather_source):
    normal_avg_temps = tmy3_weather_source.get_average_temperature(consumption_history_one_summer_electricity,electricity,"degF")
    assert abs(normal_avg_temps[0] - 68.411913043478265) < EPSILON
    assert abs(normal_avg_temps[1] - 73.327545582047691) < EPSILON
    assert abs(normal_avg_temps[2] - 74.593604488078540) < EPSILON
