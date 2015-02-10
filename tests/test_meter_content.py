from eemeter.config.yaml_parser import load
from eemeter.consumption import ConsumptionHistory
from eemeter.meter import BPI2400Meter

import pytest
from fixtures.consumption import consumption_history_1
from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source
from helpers import arrays_similar

from datetime import datetime

EPSILON = 1e-5

@pytest.mark.slow
def test_temperature_sensitivity_parameter_optimization(
        consumption_history_1,gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
            fuel_unit_str: "kWh",
            fuel_type: "electricity",
            temperature_unit_str: "degF",
            model: !obj:eemeter.models.DoubleBalancePointModel {
                x0: [1.,1.,1.,60.,7.],
                bounds: [[0,200],[0,200],[0,2000],[55,65],[2,12]],
            }
        }
        """
    meter = load(meter_yaml)

    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_source=gsod_722880_2012_2014_weather_source)

    assert arrays_similar(result['temp_sensitivity_params'],
                          [0.013456,0.029507,8.341199,65,3.8])

@pytest.mark.slow
def test_weather_normalization(consumption_history_1,
                               gsod_722880_2012_2014_weather_source,
                               tmy3_722880_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                    fuel_unit_str: "kWh",
                    fuel_type: "electricity",
                    temperature_unit_str: "degF",
                    model: !obj:eemeter.models.DoubleBalancePointModel &model {
                        x0: [1.,1.,1.,60.,7.],
                        bounds: [[0,200],[0,200],[0,2000],[55,65],[2,12]],
                    }
                },
                !obj:eemeter.meter.AnnualizedUsageMeter {
                    temperature_unit_str: "degF",
                    model: *model
                }
            ]
        }
        """
    meter = load(meter_yaml)

    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)

    assert arrays_similar(result['temp_sensitivity_params'],
                          [0.013456,0.029507,8.341199,65,3.8])

    assert abs(result['annualized_usage'] - 3087.8412641) < EPSILON

@pytest.mark.slow
def test_pre_post_parameters(consumption_history_1,
                             gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.PrePostMeter {
            splittable_args: ["consumption_history"],
            meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                fuel_unit_str: "kWh",
                fuel_type: "electricity",
                temperature_unit_str: "degF",
                model: !obj:eemeter.models.DoubleBalancePointModel {
                    x0: [1.,1.,1.,60.,7.],
                    bounds: [[0,200],[0,200],[0,2000],[55,65],[2,12]],
                }
            },
        }
        """
    meter = load(meter_yaml)

    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            retrofit_start_date=datetime(2013,9,25),
                            retrofit_end_date=datetime(2013,9,25))

    assert arrays_similar(result["temp_sensitivity_params_pre"],
            [0.016883,0.042749,6.013131,65,3.3])
    assert arrays_similar(result["temp_sensitivity_params_post"],
            [0.059923,0.001983,11.129708,55,2])

    assert isinstance(result["consumption_history_pre"],ConsumptionHistory)
    assert isinstance(result["consumption_history_post"],ConsumptionHistory)

@pytest.mark.slow
def test_gross_savings_metric(consumption_history_1,
                              gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.PrePostMeter {
                    splittable_args: ["consumption_history"],
                    meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                        fuel_unit_str: "kWh",
                        fuel_type: "electricity",
                        temperature_unit_str: "degF",
                        model: !obj:eemeter.models.DoubleBalancePointModel &model {
                            x0: [1.,1.,1.,60.,7.],
                            bounds: [[0,200],[0,200],[0,2000],[55,65],[2,12]],
                        }
                    },
                },
                !obj:eemeter.meter.GrossSavingsMeter {
                    fuel_unit_str: "kWh",
                    fuel_type: "electricity",
                    temperature_unit_str: "degF",
                    model: *model,
                }
            ]
        }
        """
    meter = load(meter_yaml)

    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            retrofit_start_date=datetime(2013,9,25),
                            retrofit_end_date=datetime(2013,9,25))

    assert arrays_similar(result["temp_sensitivity_params_pre"],
            [0.016883,0.042749,6.013131,65,3.3])
    assert arrays_similar(result["temp_sensitivity_params_post"],
            [0.059923,0.001983,11.129708,55,2])

    assert isinstance(result["consumption_history_pre"],ConsumptionHistory)
    assert isinstance(result["consumption_history_post"],ConsumptionHistory)
    assert abs(result["gross_savings"] - 494.442390) < EPSILON

@pytest.mark.slow
def test_annualized_gross_savings_metric(consumption_history_1,
                                         gsod_722880_2012_2014_weather_source,
                                         tmy3_722880_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.PrePostMeter {
                    splittable_args: ["consumption_history"],
                    meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                        fuel_unit_str: "kWh",
                        fuel_type: "electricity",
                        temperature_unit_str: "degF",
                        model: !obj:eemeter.models.DoubleBalancePointModel &model {
                            x0: [1.,1.,1.,60.,7.],
                            bounds: [[0,200],[0,200],[0,2000],[55,65],[2,12]],
                        }
                    },
                },
                !obj:eemeter.meter.AnnualizedGrossSavingsMeter {
                    fuel_type: "electricity",
                    temperature_unit_str: "degF",
                    model: *model,
                }
            ]
        }
        """
    meter = load(meter_yaml)

    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source,
                            retrofit_start_date=datetime(2013,9,25),
                            retrofit_end_date=datetime(2013,9,25))

    assert arrays_similar(result["temp_sensitivity_params_pre"],
            [0.016883,0.042749,6.013131,65,3.3])
    assert arrays_similar(result["temp_sensitivity_params_post"],
            [0.059923,0.001983,11.129708,55,2])

    assert abs(result["annualized_gross_savings"] - -1822.821986) < EPSILON

def test_fuel_type_presence_meter(consumption_history_1):

    meter_yaml = """
        !obj:eemeter.meter.FuelTypePresenceMeter {
            fuel_types: [electricity,natural_gas]
        }
        """
    meter = load(meter_yaml)
    result = meter.evaluate(consumption_history=consumption_history_1)

    assert result["electricity_presence"]
    assert not result["natural_gas_presence"]

@pytest.mark.slow
def test_princeton_scorekeeping_method(consumption_history_1,
                                       gsod_722880_2012_2014_weather_source,
                                       tmy3_722880_weather_source):
    meter = load("!obj:eemeter.meter.PRISMMeter {}")
    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)

    assert result["electricity_presence"]
    assert not result["natural_gas_presence"]

def test_bpi2400(consumption_history_1,
                 tmy3_722880_weather_source):

    meter = BPI2400Meter()
    print meter.get_inputs()
    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_normal_source=tmy3_722880_weather_source)
    assert result == {}
