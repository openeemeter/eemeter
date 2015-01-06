from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas

from eemeter.meter import MetricBase
from eemeter.meter import RawAverageUsageMetric
from eemeter.meter import AverageTemperatureMetric
from eemeter.meter import WeatherNormalizedAverageUsageMetric
from eemeter.meter import HDDCDDTemperatureSensitivityParametersMetric
from eemeter.meter import TotalHDDMetric
from eemeter.meter import TotalCDDMetric

from eemeter.meter import PrePost

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
def consumption_history_long():
    c_list = [Consumption(687600000,"J",electricity,datetime(2012,9,26),datetime(2012,10,24)),
            Consumption(874800000,"J",electricity,datetime(2012,10,24),datetime(2012,11,21)),
            Consumption(1332000000,"J",electricity,datetime(2012,11,21),datetime(2012,12,27)),
            Consumption(1454400000,"J",electricity,datetime(2012,12,27),datetime(2013,1,29)),
            Consumption(1155600000,"J",electricity,datetime(2013,1,29),datetime(2013,2,26)),
            Consumption(1195200000,"J",electricity,datetime(2013,2,26),datetime(2013,3,27)),
            Consumption(1033200000,"J",electricity,datetime(2013,3,27),datetime(2013,4,25)),
            Consumption(752400000,"J",electricity,datetime(2013,4,25),datetime(2013,5,23)),
            Consumption(889200000,"J",electricity,datetime(2013,5,23),datetime(2013,6,22)),
            Consumption(3434400000,"J",electricity,datetime(2013,6,22),datetime(2013,7,26)),
            Consumption(828000000,"J",electricity,datetime(2013,7,26),datetime(2013,8,22)),
            Consumption(2217600000,"J",electricity,datetime(2013,8,22),datetime(2013,9,25)),
            Consumption(680400000,"J",electricity,datetime(2013,9,25),datetime(2013,10,23)),
            Consumption(1062000000,"J",electricity,datetime(2013,10,23),datetime(2013,11,22)),
            Consumption(1720800000,"J",electricity,datetime(2013,11,22),datetime(2013,12,27)),
            Consumption(1915200000,"J",electricity,datetime(2013,12,27),datetime(2014,1,30)),
            Consumption(1458000000,"J",electricity,datetime(2014,1,30),datetime(2014,2,27)),
            Consumption(1332000000,"J",electricity,datetime(2014,2,27),datetime(2014,3,29)),
            Consumption(954000000,"J",electricity,datetime(2014,3,29),datetime(2014,4,26)),
            Consumption(842400000,"J",electricity,datetime(2014,4,26),datetime(2014,5,28)),
            Consumption(1220400000,"J",electricity,datetime(2014,5,28),datetime(2014,6,25)),
            Consumption(1702800000,"J",electricity,datetime(2014,6,25),datetime(2014,7,25)),
            Consumption(1375200000,"J",electricity,datetime(2014,7,25),datetime(2014,8,23)),
            Consumption(1623600000,"J",electricity,datetime(2014,8,23),datetime(2014,9,25))]


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
def gsod_722880_2012_weather_source():
    return GSODWeatherSource('722880',start_year=2012,end_year=2012)

@pytest.fixture
def tmy3_722880_2012_weather_source():
    return TMY3WeatherSource('722880',os.environ.get("TMY3_DIRECTORY"))

@pytest.fixture(params=[(datetime(2011,1,1),datetime(2014,1,1),float("nan"),float("nan")),
                        (datetime(2011,1,1),datetime(2012,7,15),float("nan"),1100),
                        (datetime(2012,7,1),datetime(2012,7,15),1250,1100),
                        (datetime(2012,7,15),datetime(2012,7,15),1250,1100),
                        (datetime(2012,7,15),datetime(2014,1,1),1250,float("nan")),])
def retrofits(request):
    return request.param

##### Tests #####

def test_metric_base():
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
@pytest.mark.internet
def test_average_temperature_metric(consumption_history_one_year_electricity,
                                    gsod_722880_2012_weather_source):
    metric = AverageTemperatureMetric(electricity)
    avg_temp = metric.evaluate(consumption_history_one_year_electricity,
                               gsod_722880_2012_weather_source)
    assert abs(avg_temp - 64.522521937955744) < EPSILON

@pytest.mark.slow
@pytest.mark.internet
def test_weather_normalize(consumption_history_one_summer_electricity,
                           gsod_722880_2012_weather_source,
                           tmy3_722880_2012_weather_source):
    metric = WeatherNormalizedAverageUsageMetric("kWh",electricity)
    result = metric.evaluate(consumption_history_one_summer_electricity,
                             np.array([1,1,100,60,5]),
                             tmy3_722880_2012_weather_source)
    assert abs(result - 37791.620062) < EPSILON

def test_hdd_cdd_temperature_sensitivity_parameters_metric(consumption_history_one_year_electricity,gsod_722880_2012_weather_source):
    metric = HDDCDDTemperatureSensitivityParametersMetric("kWh",fuel_type=electricity)
    params = metric.evaluate(consumption_history_one_year_electricity,gsod_722880_2012_weather_source)
    ts_low,ts_high,base_load,bp_low,bp_diff = params
    assert abs(ts_low - 0) < EPSILON
    assert abs(ts_high - 11.0687567 ) < EPSILON
    assert abs(base_load - 1111.103779) < EPSILON
    assert abs(bp_low - 55) < EPSILON
    assert abs(bp_diff - 2) < EPSILON

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

def test_pre_post_metric():
    assert issubclass(PrePost,MetricBase)

def test_pre_post_raw_average_usage_metric(consumption_history_one_year_electricity,retrofits):
    retrofit_start, retrofit_end, average_pre, average_post = retrofits
    metric = PrePost(RawAverageUsageMetric("kWh",fuel_type=electricity))
    result = metric.evaluate(consumption_history_one_year_electricity,retrofit_start=retrofit_start,retrofit_end=retrofit_end)
    if np.isnan(average_pre):
        assert np.isnan(result["pre"])
    else:
        assert abs(result["pre"] - average_pre) < EPSILON
    if np.isnan(average_post):
        assert np.isnan(result["post"])
    else:
        assert abs(result["post"] - average_post) < EPSILON
