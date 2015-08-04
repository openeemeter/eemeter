from eemeter.config.yaml_parser import load

from eemeter.meter import MeetsThresholds
from eemeter.meter import EstimatedReadingConsolidationMeter

from eemeter.meter import DataCollection

from eemeter.consumption import ConsumptionData

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from datetime import datetime
from datetime import timedelta

from numpy.testing import assert_allclose
import numpy as np

RTOL=1e-2
ATOL=1e-2

import pytest

@pytest.fixture(params=[
    [{"start": datetime(2012,1,1),"value": 0},
     {"start": datetime(2012,1,2), "value": 0, "estimated": True},
     {"start": datetime(2012,1,3), "value": 0},
     {"start": datetime(2012,1,4), "value": 0},
     {"start": datetime(2012,1,5), "value": np.nan}],
    [{"start": datetime(2012,1,1),"value": 0},
     {"start": datetime(2012,1,2), "value": 0},
     {"start": datetime(2012,1,4), "value": 0},
     {"start": datetime(2012,1,5), "value": np.nan}],
    [{"start": datetime(2012,1,1),"value": 0},
     {"start": datetime(2012,1,2), "value": 0},
     {"start": datetime(2012,1,4), "value": 0},
     {"start": datetime(2012,1,5), "value": np.nan},
     {"start": datetime(2012,1,6), "value": np.nan, "estimated": True}],
    [{"start": datetime(2012,1,1),"value": 0},
     {"start": datetime(2012,1,2), "value": 0},
     {"start": datetime(2012,1,4), "value": 0},
     {"start": datetime(2012,1,5), "value": np.nan},
     {"start": datetime(2012,1,6), "value": np.nan, "estimated": True},
     {"start": datetime(2012,1,7), "value": np.nan, "estimated": True}],
    [{"start": datetime(2012,1,1),"value": 0},
     {"start": datetime(2012,2,1), "value": 0, "estimated": True},
     {"start": datetime(2012,2,10), "value": 0, "estimated": True},
     {"start": datetime(2012,3,1), "value": 0},
     {"start": datetime(2012,4,1), "value": 0},
     {"start": datetime(2012,5,1), "value": np.nan}],
    [{"start": datetime(2012,1,1),"value": 0, "estimated": True},
     {"start": datetime(2012,1,10), "value": 0},
     {"start": datetime(2012,2,1), "value": 0},
     {"start": datetime(2012,4,1), "value": 0},
     {"start": datetime(2012,5,1), "value": np.nan}],
    ])
def consumption_data(request):
    return ConsumptionData(request.param, "electricity", "kWh", record_type="arbitrary_start")

def test_meets_thresholds():
    meter_yaml = """
        !obj:eemeter.meter.MeetsThresholds {
            input_mapping: {
                "one": {},
                "two": {},
                "three": {},
                "four": {},
            },
            equations: [
                ["one",   "<",   1,      0,  0, "one_lt_zero"],
                ["two",  "<=",  1,  "two",  0, "two_lte_two"],
                ["three", ">",  .5, "four",  0, "three_gt_half_four"],
                ["four", ">=",  "two",      3,  0, "four_gte_twice_three"],
                ["four", ">=",  2,      1, "two", "four_gte_four"],
            ],
            output_mapping: {
                "one_lt_zero": {},
                "two_lte_two": {},
                "three_gt_half_four": {},
                "four_gte_twice_three": {},
                "four_gte_four": {},
            },
        }
    """
    meter = load(meter_yaml)
    data_collection = DataCollection(one=1, two=2, three=3.0, four=4)
    result = meter.evaluate(data_collection)
    assert not result.get_data("one_lt_zero").value
    assert result.get_data("two_lte_two").value
    assert result.get_data("three_gt_half_four").value
    assert not result.get_data("four_gte_twice_three").value
    assert result.get_data("four_gte_four").value

def test_estimated_reading_consolidation_meter_single_fuel_type(consumption_data):

    meter_yaml = """
        !obj:eemeter.meter.EstimatedReadingConsolidationMeter {
            input_mapping: {"consumption_data": {}},
            output_mapping: {"consumption_data_no_estimated": {}},
        }
    """
    meter = load(meter_yaml)
    data_collection = DataCollection(consumption_data=consumption_data)
    result = meter.evaluate(data_collection)
    values = result.get_data("consumption_data_no_estimated").value.data.values

    assert_allclose(values, np.array([0, 0, 0, np.nan]), rtol=RTOL, atol=ATOL)

def test_debug_meter():

    meter_yaml="""
        !obj:eemeter.meter.Debug {
        }
        """
    meter = load(meter_yaml)
