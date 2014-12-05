from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas

from eemeter.meter import MetricBase
from eemeter.meter import RawAverageUsageMetric
from eemeter.meter import TemperatureRegressionParametersMetric
from eemeter.meter import AverageTemperatureMetric

from eemeter.meter import Meter
from eemeter.meter import MeterRun

from eemeter.meter import FuelTypePresenceFlag
from eemeter.meter import TimeRangePresenceFlag
from eemeter.meter import OverlappingTimePeriodsFlag
from eemeter.meter import MissingTimePeriodsFlag
from eemeter.meter import TooManyEstimatedPeriodsFlag
from eemeter.meter import InsufficientTimeRangeFlag

from eemeter.weather import GSODWeatherGetter

from datetime import datetime
import numpy as np

import pytest

EPSILON = 1e-6

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
def consumption_history_overlapping_1():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,1,1),datetime(2011,3,1)),
            Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_overlapping_2():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,1)),
            Consumption(0,"thm",natural_gas,datetime(2011,1,1),datetime(2011,3,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_overlapping_3():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,2,27),datetime(2011,3,1)),
            Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,2))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_overlapping_4():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,2)),
            Consumption(0,"thm",natural_gas,datetime(2011,2,27),datetime(2011,3,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_overlapping_5():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,2)),
            Consumption(0,"thm",natural_gas,datetime(2011,2,27),datetime(2011,3,3))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_not_overlapping_1():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,1,1),datetime(2011,3,1)),
            Consumption(0,"thm",natural_gas,datetime(2011,3,1),datetime(2011,3,2))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_not_overlapping_2():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,1)),
            Consumption(0,"thm",natural_gas,datetime(2011,3,1),datetime(2011,3,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_not_overlapping_3():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,3,1),datetime(2011,3,1)),
            Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_not_overlapping_4():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,2,28)),
            Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,1))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_not_overlapping_5():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,3,1)),
            Consumption(0,"thm",natural_gas,datetime(2011,2,28),datetime(2011,2,28))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_missing_time_period_1():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,1,1),datetime(2011,1,2)),
            Consumption(0,"thm",natural_gas,datetime(2011,1,3),datetime(2011,1,4))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_missing_time_period_2():
    c_list = [Consumption(0,"thm",natural_gas,datetime(2011,1,3),datetime(2011,1,4)),
            Consumption(0,"thm",natural_gas,datetime(2011,1,1),datetime(2011,1,2))]
    return ConsumptionHistory(c_list)

@pytest.fixture
def consumption_history_three_estimated():
    c_list = [Consumption(900,"thm",natural_gas,datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(940,"thm",natural_gas,datetime(2012,2,1),datetime(2012,3,1)),
            Consumption(800,"thm",natural_gas,datetime(2012,3,1),datetime(2012,4,1),estimated=True),
            Consumption(700,"thm",natural_gas,datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(500,"thm",natural_gas,datetime(2012,5,1),datetime(2012,6,1),estimated=True),
            Consumption(125,"thm",natural_gas,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(75,"thm",natural_gas,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(100,"thm",natural_gas,datetime(2012,8,1),datetime(2012,9,1),estimated=True),
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
def gsod_722874_2012_weather_getter():
    return GSODWeatherGetter('722874-93134',start_year=2012,end_year=2012)


##### Tests #####

def test_base_metric():
    metric = MetricBase()
    with pytest.raises(NotImplementedError):
        metric.evaluate_fuel_type([])
    assert not metric.is_flag()

def test_raw_average_usage_metric(consumption_history_one_year_electricity,
                                  consumption_history_one_year_natural_gas,
                                  consumption_history_one_summer_electricity,
                                  consumption_history_one_summer_natural_gas):
    raw_avg_usage_metric = RawAverageUsageMetric("Btu")
    elec_avg_metric = RawAverageUsageMetric("kWh",fuel_type=electricity)
    gas_avg_metric = RawAverageUsageMetric("therms",fuel_type=natural_gas)
    assert issubclass(RawAverageUsageMetric,MetricBase)
    assert not elec_avg_metric.is_flag()
    assert not gas_avg_metric.is_flag()

    avg_elec_summer_usage = elec_avg_metric.evaluate(consumption_history_one_summer_electricity)
    avg_gas_summer_usage = gas_avg_metric.evaluate(consumption_history_one_summer_natural_gas)
    avg_elec_year_usage = elec_avg_metric.evaluate(consumption_history_one_year_electricity)
    avg_gas_year_usage = gas_avg_metric.evaluate(consumption_history_one_year_natural_gas)
    assert abs(avg_elec_summer_usage - 1700) < EPSILON
    assert abs(avg_gas_summer_usage - 100) < EPSILON
    assert abs(avg_elec_year_usage - 1200) < EPSILON
    assert abs(avg_gas_year_usage - 520) < EPSILON

    avg_gas_year_usage_none = gas_avg_metric.evaluate(consumption_history_one_year_electricity)
    avg_elec_year_usage_none = elec_avg_metric.evaluate(consumption_history_one_year_natural_gas)
    avg_gas_summer_usage_none = gas_avg_metric.evaluate(consumption_history_one_summer_electricity)
    avg_elec_summer_usage_none = elec_avg_metric.evaluate(consumption_history_one_summer_natural_gas)
    assert np.isnan(avg_gas_year_usage_none)
    assert np.isnan(avg_elec_year_usage_none)
    assert np.isnan(avg_gas_summer_usage_none)
    assert np.isnan(avg_elec_summer_usage_none)

@pytest.mark.slow
def test_temperature_regression_parameters_metric(consumption_history_one_year_electricity,gsod_722874_2012_weather_getter):
    metric = TemperatureRegressionParametersMetric("kWh",electricity)
    params = metric.evaluate(consumption_history_one_year_electricity,gsod_722874_2012_weather_getter)

    class MyMeter(Meter):
        temperature_regression_metric = TemperatureRegressionParametersMetric("kWh",electricity)

    meter = MyMeter()
    results = meter.run(consumption_history=consumption_history_one_year_electricity,
                        weather_getter=gsod_722874_2012_weather_getter)

    assert results.temperature_regression_metric is not None

@pytest.mark.slow
def test_average_temperature_metric(consumption_history_one_year_electricity,gsod_722874_2012_weather_getter):
    metric = AverageTemperatureMetric(electricity)
    avg_temp = metric.evaluate(consumption_history_one_year_electricity,gsod_722874_2012_weather_getter)
    assert abs(avg_temp - 63.82780404) < EPSILON

def test_fueltype_presence_flag(consumption_history_one_year_electricity,
                                consumption_history_one_year_natural_gas):
    elec_data_present = FuelTypePresenceFlag(electricity)
    gas_data_present = FuelTypePresenceFlag(natural_gas)

    assert elec_data_present.evaluate(consumption_history_one_year_electricity)
    assert not elec_data_present.evaluate(consumption_history_one_year_natural_gas)
    assert not gas_data_present.evaluate(consumption_history_one_year_electricity)
    assert gas_data_present.evaluate(consumption_history_one_year_natural_gas)

def test_none_in_time_range_presence_flag(consumption_history_one_year_electricity):
    past_time_range_flag = TimeRangePresenceFlag(datetime(1900,1,1),datetime(1950,1,1))
    future_time_range_flag = TimeRangePresenceFlag(datetime(2050,1,1),datetime(2100,1,1))
    recent_time_range_flag = TimeRangePresenceFlag(datetime(2012,1,1),datetime(2013,1,1))

    elec_past_time_range_flag = TimeRangePresenceFlag(datetime(1900,1,1),datetime(1950,1,1),fuel_type=electricity)
    elec_future_time_range_flag = TimeRangePresenceFlag(datetime(2050,1,1),datetime(2100,1,1),fuel_type=electricity)
    elec_recent_time_range_flag = TimeRangePresenceFlag(datetime(2012,1,1),datetime(2013,1,1),fuel_type=electricity)

    gas_past_time_range_flag = TimeRangePresenceFlag(datetime(1900,1,1),datetime(1950,1,1),fuel_type=natural_gas)
    gas_future_time_range_flag = TimeRangePresenceFlag(datetime(2050,1,1),datetime(2100,1,1),fuel_type=natural_gas)
    gas_recent_time_range_flag = TimeRangePresenceFlag(datetime(2012,1,1),datetime(2013,1,1),fuel_type=natural_gas)

    assert not past_time_range_flag.evaluate(consumption_history_one_year_electricity)["electricity"]
    assert not future_time_range_flag.evaluate(consumption_history_one_year_electricity)["electricity"]
    assert recent_time_range_flag.evaluate(consumption_history_one_year_electricity)["electricity"]

    with pytest.raises(KeyError):
        assert not past_time_range_flag.evaluate(consumption_history_one_year_electricity)["natural_gas"]

    with pytest.raises(KeyError):
        assert not future_time_range_flag.evaluate(consumption_history_one_year_electricity)["natural_gas"]

    with pytest.raises(KeyError):
        assert recent_time_range_flag.evaluate(consumption_history_one_year_electricity)["natural_gas"]

    assert not elec_past_time_range_flag.evaluate(consumption_history_one_year_electricity)
    assert not elec_future_time_range_flag.evaluate(consumption_history_one_year_electricity)
    assert elec_recent_time_range_flag.evaluate(consumption_history_one_year_electricity)

    assert not gas_past_time_range_flag.evaluate(consumption_history_one_year_electricity)
    assert not gas_future_time_range_flag.evaluate(consumption_history_one_year_electricity)
    assert not gas_recent_time_range_flag.evaluate(consumption_history_one_year_electricity)

def test_overlapping_periods_flag(consumption_history_one_year_electricity,
                                  consumption_history_overlapping_1,
                                  consumption_history_overlapping_2,
                                  consumption_history_overlapping_3,
                                  consumption_history_overlapping_4,
                                  consumption_history_overlapping_5,
                                  consumption_history_not_overlapping_1,
                                  consumption_history_not_overlapping_2,
                                  consumption_history_not_overlapping_3,
                                  consumption_history_not_overlapping_4,
                                  consumption_history_not_overlapping_5):

    overlap_flag = OverlappingTimePeriodsFlag()
    elec_overlap_flag = OverlappingTimePeriodsFlag(fuel_type=electricity)
    gas_overlap_flag = OverlappingTimePeriodsFlag(fuel_type=natural_gas)

    assert not overlap_flag.evaluate(consumption_history_one_year_electricity)["electricity"]
    assert overlap_flag.evaluate(consumption_history_overlapping_1)["natural_gas"]
    assert overlap_flag.evaluate(consumption_history_overlapping_2)["natural_gas"]
    assert overlap_flag.evaluate(consumption_history_overlapping_3)["natural_gas"]
    assert overlap_flag.evaluate(consumption_history_overlapping_4)["natural_gas"]
    assert overlap_flag.evaluate(consumption_history_overlapping_5)["natural_gas"]
    assert not overlap_flag.evaluate(consumption_history_not_overlapping_1)["natural_gas"]
    assert not overlap_flag.evaluate(consumption_history_not_overlapping_2)["natural_gas"]
    assert not overlap_flag.evaluate(consumption_history_not_overlapping_3)["natural_gas"]
    assert not overlap_flag.evaluate(consumption_history_not_overlapping_4)["natural_gas"]
    assert not overlap_flag.evaluate(consumption_history_not_overlapping_5)["natural_gas"]

    assert not elec_overlap_flag.evaluate(consumption_history_one_year_electricity)

    assert gas_overlap_flag.evaluate(consumption_history_overlapping_1)
    assert gas_overlap_flag.evaluate(consumption_history_overlapping_2)
    assert gas_overlap_flag.evaluate(consumption_history_overlapping_3)
    assert gas_overlap_flag.evaluate(consumption_history_overlapping_4)
    assert gas_overlap_flag.evaluate(consumption_history_overlapping_5)
    assert not gas_overlap_flag.evaluate(consumption_history_not_overlapping_1)
    assert not gas_overlap_flag.evaluate(consumption_history_not_overlapping_2)
    assert not gas_overlap_flag.evaluate(consumption_history_not_overlapping_3)
    assert not gas_overlap_flag.evaluate(consumption_history_not_overlapping_4)
    assert not gas_overlap_flag.evaluate(consumption_history_not_overlapping_5)

def test_missing_time_periods_flag(consumption_history_one_summer_electricity,
                                   consumption_history_missing_time_period_1,
                                   consumption_history_missing_time_period_2):
    missing_flag = MissingTimePeriodsFlag()
    elec_missing_flag = MissingTimePeriodsFlag(fuel_type=electricity)
    gas_missing_flag = MissingTimePeriodsFlag(fuel_type=natural_gas)

    assert not missing_flag.evaluate(consumption_history_one_summer_electricity)["electricity"]
    assert missing_flag.evaluate(consumption_history_missing_time_period_1)["natural_gas"]
    assert missing_flag.evaluate(consumption_history_missing_time_period_2)["natural_gas"]

    assert not elec_missing_flag.evaluate(consumption_history_one_summer_electricity)

    assert gas_missing_flag.evaluate(consumption_history_missing_time_period_1)
    assert gas_missing_flag.evaluate(consumption_history_missing_time_period_2)

def test_too_many_estimated_time_periods_flag(consumption_history_three_estimated):
    more_than_two_estimated_flag = TooManyEstimatedPeriodsFlag(2)
    more_than_three_estimated_flag = TooManyEstimatedPeriodsFlag(3)
    assert more_than_two_estimated_flag.evaluate(consumption_history_three_estimated)["natural_gas"]
    assert more_than_two_estimated_flag.evaluate(consumption_history_three_estimated)["natural_gas"]
    assert not more_than_three_estimated_flag.evaluate(consumption_history_three_estimated)["natural_gas"]

    gas_more_than_two_estimated_flag = TooManyEstimatedPeriodsFlag(2,fuel_type=natural_gas)
    gas_more_than_three_estimated_flag = TooManyEstimatedPeriodsFlag(3,fuel_type=natural_gas)
    assert gas_more_than_two_estimated_flag.evaluate(consumption_history_three_estimated)
    assert not gas_more_than_three_estimated_flag.evaluate(consumption_history_three_estimated)


def test_insufficient_time_range_flag(consumption_history_one_year_electricity):
    short_time_range_flag = InsufficientTimeRangeFlag(100)
    exact_time_range_flag = InsufficientTimeRangeFlag(365)
    long_time_range_flag = InsufficientTimeRangeFlag(1000)
    assert not short_time_range_flag.evaluate(consumption_history_one_year_electricity)["electricity"]
    assert not exact_time_range_flag.evaluate(consumption_history_one_year_electricity)["electricity"]
    assert long_time_range_flag.evaluate(consumption_history_one_year_electricity)["electricity"]

    elec_short_time_range_flag = InsufficientTimeRangeFlag(100,fuel_type=electricity)
    elec_exact_time_range_flag = InsufficientTimeRangeFlag(365,fuel_type=electricity)
    elec_long_time_range_flag = InsufficientTimeRangeFlag(1000,fuel_type=electricity)
    assert not elec_short_time_range_flag.evaluate(consumption_history_one_year_electricity)
    assert not elec_exact_time_range_flag.evaluate(consumption_history_one_year_electricity)
    assert elec_long_time_range_flag.evaluate(consumption_history_one_year_electricity)

def test_meter_run(meter_run_simple):
    assert meter_run_simple.elec_avg_usage == 100

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
