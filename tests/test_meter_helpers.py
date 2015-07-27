from eemeter.config.yaml_parser import load

from eemeter.meter import MeetsThresholds
from eemeter.meter import EstimatedReadingConsolidationMeter

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
    meter = MeetsThresholds(values=["one","two","three","four"],
                            thresholds=[0,"two","four",3],
                            operations=["lt","lte","gt","gte"],
                            proportions=[1,1,.5,2],
                            output_names=["one_lt_zero","two_lte_two","three_gt_half_two","four_gte_twice_three"])
    result = meter.evaluate(one=1,two=2,three=3,four=4)
    assert not result["one_lt_zero"]
    assert result["two_lte_two"]
    assert result["three_gt_half_two"]
    assert not result["four_gte_twice_three"]

def test_estimated_reading_consolidation_meter_single_fuel_type(consumption_data):

    meter = EstimatedReadingConsolidationMeter()
    result = meter.evaluate(consumption_data=consumption_data)
    values = result["consumption_data_no_estimated"].data.values

    assert_allclose(values,np.array([0,0,0,np.nan]),rtol=RTOL,atol=ATOL)

def test_debug_meter():

    meter_yaml="""
        !obj:eemeter.meter.Debug {
        }
        """
    meter = load(meter_yaml)

def test_dummy_meter():

    meter_yaml="""
        !obj:eemeter.meter.DummyMeter {
        }
        """
    meter = load(meter_yaml)
