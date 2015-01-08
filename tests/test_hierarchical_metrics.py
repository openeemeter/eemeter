from eemeter.config.yaml_parser import load
from fixtures.consumption import consumption_history_1
from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from helpers import arrays_similar
import pytest

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
                    fuel_unit_str: "kWh",
                    fuel_type: "electricity",
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
