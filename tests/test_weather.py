from eemeter.weather import WeatherGetterBase
from eemeter.weather import GSODWeatherGetter
from eemeter.weather import WeatherUndergroundWeatherGetter

from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import Consumption
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas

from datetime import datetime
import pytest
import os

EPSILON = 10e-6

@pytest.fixture
def consumption_history_one_summer_electricity():
    c_list = [Consumption(1600,"kWh",electricity,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(1700,"kWh",electricity,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(1800,"kWh",electricity,datetime(2012,8,1),datetime(2012,9,1))]
    return ConsumptionHistory(c_list)

def test_weather_getter_base(consumption_history_one_summer_electricity):
    weather_getter = WeatherGetterBase()
    with pytest.raises(NotImplementedError):
        hdd = weather_getter.get_average_temperature(consumption_history_one_summer_electricity,electricity)

@pytest.mark.slow
def test_gsod_weather_getter(consumption_history_one_summer_electricity):
    gsod_weather_getter = GSODWeatherGetter('722874-93134',start_year=2012,end_year=2012)
    avg_temps = gsod_weather_getter.get_average_temperature(consumption_history_one_summer_electricity,electricity)
    assert abs(avg_temps[0] - 66.3833333333) < EPSILON
    assert abs(avg_temps[1] - 67.8032258065) < EPSILON
    assert abs(avg_temps[2] - 74.4451612903) < EPSILON

@pytest.mark.slow
def test_weather_underground_weather_getter(consumption_history_one_summer_electricity):
    wunderground_api_key = os.environ.get('WEATHERUNDERGROUND_API_KEY')
    if wunderground_api_key:
        wu_weather_getter = WeatherUndergroundWeatherGetter('60605',
                                                            datetime(2012,6,1),
                                                            datetime(2012,10,1),
                                                            wunderground_api_key)
        avg_temps = wu_weather_getter.get_average_temperature(consumption_history_one_summer_electricity,electricity)
        assert abs(avg_temps[0] - 74.4333333333) < EPSILON
        assert abs(avg_temps[1] - 82.6774193548) < EPSILON
        assert abs(avg_temps[2] - 75.4516129032) < EPSILON
    else:
        warning.warn("Skipping WeatherUndergroundWeatherGetter tests. "
                     "Please set the environment variable "
                     "WEATHERUNDERGOUND_API_KEY to run the tests.")
