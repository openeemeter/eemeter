from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas

from eemeter.meter import RawAverageUsageMetric
from eemeter.meter import AverageTemperatureMetric
from eemeter.meter import WeatherNormalizedAverageUsageMetric
from eemeter.meter import HDDCDDTemperatureSensitivityParametersMetric
from eemeter.meter import TotalHDDMetric
from eemeter.meter import TotalCDDMetric

from eemeter.meter import PrePost

from eemeter.meter import Meter
from eemeter.meter import MeterRun

from eemeter.meter import FuelTypePresenceFlag
from eemeter.meter import TimeRangePresenceFlag
from eemeter.meter import OverlappingTimePeriodsFlag
from eemeter.meter import MissingTimePeriodsFlag
from eemeter.meter import TooManyEstimatedPeriodsFlag
from eemeter.meter import InsufficientTimeRangeFlag

from eemeter.weather import GSODWeatherSource
from eemeter.weather import TMY3WeatherSource

from datetime import datetime
import os

import numpy as np

import pytest

EPSILON = 1e-6

##### Fixtures #####

@pytest.fixture
def consumption_history_one_year_electricity():
    c_list = [Consumption(1000,"kWh",electricity,datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(1100,"kWh",electricity,datetime(2012,2,1),datetime(2012,3,1)),
            Consumption(1200,"kWh",electricity,datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(1300,"kWh",electricity,datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(1400,"kWh",electricity,datetime(2012,5,1),datetime(2012,6,1)),
            Consumption(1500,"kWh",electricity,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(1400,"kWh",electricity,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(1300,"kWh",electricity,datetime(2012,8,1),datetime(2012,9,1)),
            Consumption(1200,"kWh",electricity,datetime(2012,9,1),datetime(2012,10,1)),
            Consumption(1100,"kWh",electricity,datetime(2012,10,1),datetime(2012,11,1)),
            Consumption(1000,"kWh",electricity,datetime(2012,11,1),datetime(2012,12,1)),
            Consumption(900,"kWh",electricity,datetime(2012,12,1),datetime(2013,1,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_one_year_natural_gas():
    c_list = [Consumption(900,"thm",natural_gas,datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(940,"thm",natural_gas,datetime(2012,2,1),datetime(2012,3,1)),
            Consumption(800,"thm",natural_gas,datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(700,"thm",natural_gas,datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(500,"thm",natural_gas,datetime(2012,5,1),datetime(2012,6,1)),
            Consumption(125,"thm",natural_gas,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(75,"thm",natural_gas,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(100,"thm",natural_gas,datetime(2012,8,1),datetime(2012,9,1)),
            Consumption(200,"thm",natural_gas,datetime(2012,9,1),datetime(2012,10,1)),
            Consumption(400,"thm",natural_gas,datetime(2012,10,1),datetime(2012,11,1)),
            Consumption(700,"thm",natural_gas,datetime(2012,11,1),datetime(2012,12,1)),
            Consumption(800,"thm",natural_gas,datetime(2012,12,1),datetime(2013,1,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def metric_list():
    elec_avg_metric = RawAverageUsageMetric("kWh",fuel_type = electricity)
    gas_avg_metric = RawAverageUsageMetric("therms",fuel_type = natural_gas)
    metrics = [elec_avg_metric,gas_avg_metric]
    return metrics

@pytest.fixture
def meter_run_simple():
    data = {"elec_avg_usage": 100}
    return MeterRun(data)

@pytest.fixture(scope="module")
def gsod_722880_2012_weather_source():
    return GSODWeatherSource('722880',start_year=2012,end_year=2012)

@pytest.fixture
def tmy3_722880_2012_weather_source():
    return TMY3WeatherSource('722880',os.environ.get("TMY3_DIRECTORY"))

##### Tests #####

def test_meter_run(meter_run_simple):
    assert meter_run_simple.elec_avg_usage == 100
    assert meter_run_simple #test __nonzero__

def test_meter_class_integration(metric_list,consumption_history_one_year_electricity):
    class MyMeter(Meter):
        elec_avg_usage = RawAverageUsageMetric("kWh",fuel_type=electricity)
        gas_avg_usage = RawAverageUsageMetric("therms",fuel_type=natural_gas)
        elec_data_present = FuelTypePresenceFlag(electricity)
        gas_data_present = FuelTypePresenceFlag(natural_gas)

    assert isinstance(MyMeter.metrics["elec_avg_usage"],RawAverageUsageMetric)
    assert isinstance(MyMeter.metrics["gas_avg_usage"],RawAverageUsageMetric)

    # Meter instantiation
    meter = MyMeter()
    assert "elec_avg_usage" in meter.metrics.keys()
    assert "gas_avg_usage" in meter.metrics.keys()

    # Meter evaluation
    result = meter.run(consumption_history=consumption_history_one_year_electricity)
    assert isinstance(result,MeterRun)
    assert "MeterRun" in str(result)

    # Meter.run function only takes keyword args
    with pytest.raises(TypeError):
        result = meter.run(consumption_history_one_year_electricity)

    # Meter checking
    assert abs(result.elec_avg_usage - 1200) < EPSILON
    assert np.isnan(result.gas_avg_usage)
    assert result.elec_data_present
    assert not result.gas_data_present

@pytest.mark.slow
def test_meter_stages(consumption_history_one_year_electricity,
                      gsod_722880_2012_weather_source,
                      tmy3_722880_2012_weather_source):

    # create a staged meter
    class MyMeter(Meter):
        stages = [
                {"temperature_sensitivity_parameters": HDDCDDTemperatureSensitivityParametersMetric("kWh",fuel_type=electricity)},
                {"total_heating_degree_days":TotalHDDMetric(fuel_type=electricity),
                 "total_cooling_degree_days":TotalCDDMetric(fuel_type=electricity)},
                {"normalized_usage": WeatherNormalizedAverageUsageMetric("kWh",fuel_type=electricity)}]
    meter = MyMeter()
    result = meter.run(consumption_history=consumption_history_one_year_electricity,
            weather_source=gsod_722880_2012_weather_source,
            weather_normal_source=tmy3_722880_2012_weather_source)
    assert abs(result.total_heating_degree_days - 231.1) < EPSILON
    assert abs(result.total_cooling_degree_days - 3149.3) < EPSILON
    assert abs(result.normalized_usage - 438695.048255) < EPSILON
