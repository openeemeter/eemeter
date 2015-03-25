from eemeter.config.yaml_parser import load

from eemeter.meter import MeetsThresholds
from eemeter.meter import EstimatedReadingConsolidationMeter

from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory

from eemeter.generator import ConsumptionGenerator
from eemeter.generator import generate_periods

from fixtures.consumption import generated_consumption_history_pre_post_1

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from datetime import datetime
from datetime import timedelta

from numpy.testing import assert_allclose

RTOL=1e-2
ATOL=1e-2

import pytest

@pytest.mark.slow
def test_pre_post_parameters(generated_consumption_history_pre_post_1,
                             gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.PrePost {
            splittable_args: ["consumption_history"],
            pre_meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter &meter {
                fuel_unit_str: "kWh",
                fuel_type: "electricity",
                temperature_unit_str: "degF",
                model: !obj:eemeter.models.TemperatureSensitivityModel {
                    cooling: True,
                    heating: True,
                    initial_params: {
                        base_consumption: 0,
                        heating_slope: 0,
                        cooling_slope: 0,
                        heating_reference_temperature: 60,
                        cooling_reference_temperature: 70,
                    },
                    param_bounds: {
                        base_consumption: [0,2000],
                        heating_slope: [0,200],
                        cooling_slope: [0,200],
                        heating_reference_temperature: [55,65],
                        cooling_reference_temperature: [65,75],
                    },
                },
            },
            post_meter: *meter,
        }
        """
    meter = load(meter_yaml)

    ch, pre_params, post_params, retrofit = generated_consumption_history_pre_post_1

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            retrofit_start_date=retrofit,
                            retrofit_completion_date=retrofit)

    assert_allclose(result['temp_sensitivity_params_pre'], pre_params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['temp_sensitivity_params_post'], post_params, rtol=RTOL, atol=ATOL)

    assert isinstance(result["consumption_history_pre"],ConsumptionHistory)
    assert isinstance(result["consumption_history_post"],ConsumptionHistory)

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

def test_estimated_reading_consolidation_meter_single_fuel_type():
    ch1 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,3,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch2 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])


    ch3 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"kWh","electricity",datetime(2012,5,1),datetime(2012,6,1),estimated=True)
            ])

    ch4 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"kWh","electricity",datetime(2012,5,1),datetime(2012,6,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,6,1),datetime(2012,7,1),estimated=True)
            ])

    ch5 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,2,10),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,2,10),datetime(2012,3,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch6 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,1,10),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,1,10),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch7 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,1,10),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,1,10),datetime(2012,2,1))
            ])

    ch_no_est = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    meter = EstimatedReadingConsolidationMeter()
    result1 = meter.evaluate(consumption_history=ch1)
    result2 = meter.evaluate(consumption_history=ch2)
    result3 = meter.evaluate(consumption_history=ch3)
    result4 = meter.evaluate(consumption_history=ch4)
    result5 = meter.evaluate(consumption_history=ch5)
    result6 = meter.evaluate(consumption_history=ch6)
    result7 = meter.evaluate(consumption_history=ch7)

    for result in [result1,result2,result3,result4,result5,result6,result7]:
        for c1,c2 in zip(result["consumption_history_no_estimated"].iteritems(),
                         ch_no_est.iteritems()):
            assert c1 == c2

def test_estimated_reading_consolidation_meter_multiple_fuel_type():
    ch1 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,3,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,2,1),datetime(2012,3,1),estimated=True),
            Consumption(0,"therm","natural_gas",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch_no_est = ConsumptionHistory([
            Consumption(0,"therm","natural_gas",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,4,1),datetime(2012,5,1))
            ])

    meter = EstimatedReadingConsolidationMeter()
    result1 = meter.evaluate(consumption_history=ch1)

    for result in [result1]:
        for c1,c2 in zip(result["consumption_history_no_estimated"].get("electricity"),
                         ch_no_est.get("electricity")):
            assert c1 == c2
        for c1,c2 in zip(result["consumption_history_no_estimated"].get("natural_gas"),
                         ch_no_est.get("natural_gas")):
            assert c1 == c2

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
