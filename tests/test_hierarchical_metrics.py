from eemeter.config.yaml_parser import load
from eemeter.consumption import ConsumptionHistory
from fixtures.consumption import consumption_history_1
from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from helpers import arrays_similar
import pytest

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
                          [0.323828,1.316172,9.524184,65.,6.075])

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
                          [0.323828,1.316172,9.524184,65.,6.075])

    assert abs(result['annualized_usage'] - 4674.565226) < EPSILON

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

    assert arrays_similar(result["temp_sensitivity_params_pre"],[0.32440489,1.70232533,8.81616552,62.32858038,8.74641962])
    assert arrays_similar(result["temp_sensitivity_params_post"],[1.48388184,0.44307332,11.16436228,59.47276951,10.32410602])

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

    assert arrays_similar(result["temp_sensitivity_params_pre"],[0.32440489,1.70232533,8.81616552,62.32858038,8.74641962])
    assert arrays_similar(result["temp_sensitivity_params_post"],[1.48388184,0.44307332,11.16436228,59.47276951,10.32410602])

    assert isinstance(result["consumption_history_pre"],ConsumptionHistory)
    assert isinstance(result["consumption_history_post"],ConsumptionHistory)
    assert abs(result["gross_savings"] - 405.345513) < EPSILON

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

    assert arrays_similar(result["temp_sensitivity_params_pre"],[0.32440489,1.70232533,8.81616552,62.32858038,8.74641962])
    assert arrays_similar(result["temp_sensitivity_params_post"],[1.48388184,0.44307332,11.16436228,59.47276951,10.32410602])

    assert abs(result["annualized_gross_savings"] - -781.944405) < EPSILON
