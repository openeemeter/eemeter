from eemeter.config.yaml_parser import load
from eemeter.consumption import ConsumptionHistory

import pytest

from eemeter.meter import BPI2400Meter
from eemeter.models import TemperatureSensitivityModel
from eemeter.meter import AnnualizedUsageMeter

from fixtures.consumption import consumption_history_1
from fixtures.consumption import generated_consumption_history_1
from fixtures.consumption import generated_consumption_history_with_annualized_usage_1
from fixtures.consumption import generated_consumption_history_pre_post_1
from fixtures.consumption import generated_consumption_history_pre_post_with_gross_savings_1
from fixtures.consumption import generated_consumption_history_pre_post_with_annualized_gross_savings_1
from fixtures.consumption import prism_outputs_1
from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from datetime import datetime

from numpy.testing import assert_allclose

RTOL = 1e-2
ATOL = 1e-2

@pytest.mark.slow
def test_temperature_sensitivity_parameter_optimization(
        generated_consumption_history_1,gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
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
        }
        """
    meter = load(meter_yaml)

    ch, params = generated_consumption_history_1

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source)

    assert_allclose(result['temp_sensitivity_params'], params, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_weather_normalization(generated_consumption_history_with_annualized_usage_1,
                               gsod_722880_2012_2014_weather_source,
                               tmy3_722880_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            sequence: [
                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                    fuel_unit_str: "kWh",
                    fuel_type: "electricity",
                    temperature_unit_str: "degF",
                    model: !obj:eemeter.models.TemperatureSensitivityModel &model {
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
                !obj:eemeter.meter.AnnualizedUsageMeter {
                    temperature_unit_str: "degF",
                    model: *model
                }
            ]
        }
        """
    meter = load(meter_yaml)

    ch, params, annualized_usage = generated_consumption_history_with_annualized_usage_1
    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)

    assert_allclose(result['temp_sensitivity_params'], params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['annualized_usage'], annualized_usage, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_pre_post_parameters(generated_consumption_history_pre_post_1,
                             gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.PrePost {
            splittable_args: ["consumption_history"],
            meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
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
        }
        """
    meter = load(meter_yaml)

    ch, pre_params, post_params, retrofit = generated_consumption_history_pre_post_1

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            retrofit_start_date=retrofit,
                            retrofit_end_date=retrofit)

    assert_allclose(result['temp_sensitivity_params_pre'], pre_params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['temp_sensitivity_params_post'], post_params, rtol=RTOL, atol=ATOL)

    assert isinstance(result["consumption_history_pre"],ConsumptionHistory)
    assert isinstance(result["consumption_history_post"],ConsumptionHistory)

@pytest.mark.slow
def test_gross_savings_metric(generated_consumption_history_pre_post_with_gross_savings_1,
                              gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            sequence: [
                !obj:eemeter.meter.PrePost {
                    splittable_args: ["consumption_history"],
                    meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                        fuel_unit_str: "kWh",
                        fuel_type: "electricity",
                        temperature_unit_str: "degF",
                        model: !obj:eemeter.models.TemperatureSensitivityModel &model {
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

    ch, pre_params, post_params, retrofit, savings = generated_consumption_history_pre_post_with_gross_savings_1

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            retrofit_start_date=retrofit,
                            retrofit_end_date=retrofit)

    assert_allclose(result['temp_sensitivity_params_pre'], pre_params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['temp_sensitivity_params_post'], post_params, rtol=RTOL, atol=ATOL)

    assert isinstance(result["consumption_history_pre"],ConsumptionHistory)
    assert isinstance(result["consumption_history_post"],ConsumptionHistory)

    assert_allclose(result["gross_savings"],savings, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_annualized_gross_savings_metric(generated_consumption_history_pre_post_with_annualized_gross_savings_1,
                                         gsod_722880_2012_2014_weather_source,
                                         tmy3_722880_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            sequence: [
                !obj:eemeter.meter.PrePost {
                    splittable_args: ["consumption_history"],
                    meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
                        fuel_unit_str: "kWh",
                        fuel_type: "electricity",
                        temperature_unit_str: "degF",
                        model: !obj:eemeter.models.TemperatureSensitivityModel &model {
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

    ch, pre_params, post_params, retrofit, savings = generated_consumption_history_pre_post_with_annualized_gross_savings_1

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source,
                            retrofit_start_date=retrofit,
                            retrofit_end_date=retrofit)

    assert_allclose(result['temp_sensitivity_params_pre'], pre_params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['temp_sensitivity_params_post'], post_params, rtol=RTOL, atol=ATOL)

    assert isinstance(result["consumption_history_pre"],ConsumptionHistory)
    assert isinstance(result["consumption_history_post"],ConsumptionHistory)


    assert_allclose(result["annualized_gross_savings"], savings, rtol=RTOL, atol=ATOL)

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
def test_princeton_scorekeeping_method(prism_outputs_1,
                                       gsod_722880_2012_2014_weather_source,
                                       tmy3_722880_weather_source):
    meter = load("!obj:eemeter.meter.PRISMMeter {}")

    ch, elec_params, elec_presence, elec_annualized_usage, elec_error = prism_outputs_1
    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)



    assert result.get("electricity_presence") == elec_presence
    assert_allclose(result.get("temp_sensitivity_params_electricity"),
            elec_params, rtol=RTOL, atol=ATOL)
    assert_allclose(result.get("annualized_usage_electricity"),
            elec_annualized_usage, rtol=RTOL, atol=ATOL)
    assert_allclose(result.get("daily_standard_error_electricity"),
            elec_error, rtol=RTOL, atol=ATOL)

    assert not result.get("natural_gas_presence")
    assert result.get("temp_sensitivity_params_natural_gas") is None
    assert result.get("annualized_usage_natural_gas") is None
    assert result.get("daily_standard_error_natural_gas") is None

def test_bpi2400(consumption_history_1,
                 tmy3_722880_weather_source):

    meter = BPI2400Meter()
    result = meter.evaluate(consumption_history=consumption_history_1,
                            weather_normal_source=tmy3_722880_weather_source)
    assert "consumption_history_no_estimated" in result
    assert False
