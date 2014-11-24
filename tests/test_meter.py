from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas
from eemeter.meter import MetricBase
from eemeter.meter import RawAverageUsageMetric
from eemeter.meter import Meter
from eemeter.meter import MeterRun
from eemeter.meter import FuelTypePresenceFlag

from datetime import datetime
import numpy as np

import pytest

EPSILON = 10e-6

##### Fixtures #####

@pytest.fixture
def consumption_history_one_summer_electricity():
    c_list = [Consumption(1600,"kWh",electricity,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(1700,"kWh",electricity,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(1800,"kWh",electricity,datetime(2012,8,1),datetime(2012,9,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_one_summer_natural_gas():
    c_list = [Consumption(125,"thm",natural_gas,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(75,"thm",natural_gas,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(100,"thm",natural_gas,datetime(2012,8,1),datetime(2012,9,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def degrees_f_one_year():
    return [20,15,20,35,55,65,80,80,60,45,40,30]

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
    elec_avg_metric = RawAverageUsageMetric(electricity,"kWh")
    gas_avg_metric = RawAverageUsageMetric(natural_gas,"therms")
    metrics = [elec_avg_metric,gas_avg_metric]
    return metrics

@pytest.fixture
def meter_run_simple():
    data = {"elec_avg_usage": 100}
    return MeterRun(data)

##### Tests #####

def test_base_metric():
    metric = MetricBase()
    with pytest.raises(NotImplementedError):
        metric.evaluate(None)
    assert not metric.is_flag()

def test_raw_average_usage_metric(consumption_history_one_year_electricity,
                                  consumption_history_one_year_natural_gas,
                                  consumption_history_one_summer_electricity,
                                  consumption_history_one_summer_natural_gas):
    elec_avg_metric = RawAverageUsageMetric(electricity,"kWh")
    gas_avg_metric = RawAverageUsageMetric(natural_gas,"therms")
    assert issubclass(RawAverageUsageMetric,MetricBase)

    avg_elec_year_usage = elec_avg_metric.evaluate(consumption_history_one_year_electricity)
    assert abs(avg_elec_year_usage - 1200) < EPSILON

    avg_gas_year_usage = gas_avg_metric.evaluate(consumption_history_one_year_natural_gas)
    assert abs(avg_gas_year_usage - 520) < EPSILON

    avg_elec_summer_usage = elec_avg_metric.evaluate(consumption_history_one_summer_electricity)
    assert abs(avg_elec_summer_usage - 1700) < EPSILON

    avg_gas_summer_usage = gas_avg_metric.evaluate(consumption_history_one_summer_natural_gas)
    assert abs(avg_gas_summer_usage - 100) < EPSILON

    avg_gas_year_usage_none = gas_avg_metric.evaluate(consumption_history_one_year_electricity)
    assert np.isnan(avg_gas_year_usage_none)

    avg_elec_year_usage_none = elec_avg_metric.evaluate(consumption_history_one_year_natural_gas)
    assert np.isnan(avg_elec_year_usage_none)

    avg_gas_summer_usage_none = gas_avg_metric.evaluate(consumption_history_one_summer_electricity)
    assert np.isnan(avg_gas_summer_usage_none)

    avg_elec_summer_usage_none = elec_avg_metric.evaluate(consumption_history_one_summer_natural_gas)
    assert np.isnan(avg_elec_summer_usage_none)

def test_fueltype_presence_metric(consumption_history_one_year_electricity,
                                  consumption_history_one_year_natural_gas):
    elec_data_present = FuelTypePresenceFlag(electricity)
    gas_data_present = FuelTypePresenceFlag(natural_gas)

    assert elec_data_present.evaluate(consumption_history_one_year_electricity)
    assert not elec_data_present.evaluate(consumption_history_one_year_natural_gas)
    assert not gas_data_present.evaluate(consumption_history_one_year_electricity)
    assert gas_data_present.evaluate(consumption_history_one_year_natural_gas)

def test_meter_run(meter_run_simple):
    assert meter_run_simple.elec_avg_usage == 100

def test_meter_class_integration(metric_list,consumption_history_one_year_electricity):
    class MyMeter(Meter):
        elec_avg_usage = RawAverageUsageMetric(electricity,"kWh")
        gas_avg_usage = RawAverageUsageMetric(natural_gas,"therms")
        elec_data_present = FuelTypePresenceFlag(electricity)
        gas_data_present = FuelTypePresenceFlag(natural_gas)

    assert isinstance(MyMeter.metrics["elec_avg_usage"],RawAverageUsageMetric)
    assert isinstance(MyMeter.metrics["gas_avg_usage"],RawAverageUsageMetric)

    # Meter instantiation
    meter = MyMeter()
    assert "elec_avg_usage" in meter.metrics.keys()
    assert "gas_avg_usage" in meter.metrics.keys()

    # Meter evaluation
    result = meter.run(consumption_history_one_year_electricity)
    assert isinstance(result,MeterRun)

    # Meter checking
    assert abs(result.elec_avg_usage - 1200) < EPSILON
    assert np.isnan(result.gas_avg_usage)
    assert result.elec_data_present
    assert not result.gas_data_present
