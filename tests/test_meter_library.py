from eemeter.config.yaml_parser import load

from eemeter.consumption import ConsumptionHistory
from eemeter.consumption import Consumption

from eemeter.meter import TimeSpanMeter
from eemeter.meter import TotalHDDMeter
from eemeter.meter import TotalCDDMeter
from eemeter.meter import NormalAnnualHDD
from eemeter.meter import NormalAnnualCDD
from eemeter.meter import NPeriodsMeetingHDDPerDayThreshold
from eemeter.meter import NPeriodsMeetingCDDPerDayThreshold
from eemeter.meter import RecentReadingMeter
from eemeter.meter import CVRMSE
from eemeter.meter import AverageDailyUsage
from eemeter.meter import EstimatedAverageDailyUsage
from eemeter.meter import RMSE
from eemeter.meter import RSquared

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from fixtures.consumption import generated_consumption_history_1
from fixtures.consumption import generated_consumption_history_with_annualized_usage_1
from fixtures.consumption import generated_consumption_history_pre_post_with_gross_savings_1
from fixtures.consumption import generated_consumption_history_pre_post_with_annualized_gross_savings_1
from fixtures.consumption import consumption_history_1
from fixtures.consumption import time_span_1
from fixtures.consumption import generated_consumption_history_with_hdd_1
from fixtures.consumption import generated_consumption_history_with_cdd_1
from fixtures.consumption import generated_consumption_history_with_n_periods_hdd_1
from fixtures.consumption import generated_consumption_history_with_n_periods_cdd_1

from datetime import datetime
from datetime import timedelta

from numpy.testing import assert_allclose
import numpy as np

RTOL = 1e-2
ATOL = 1e-2

import pytest

@pytest.mark.slow
def test_temperature_sensitivity_parameter_optimization(
        generated_consumption_history_1,gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
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
                            weather_source=gsod_722880_2012_2014_weather_source,
                            fuel_type="electricity",
                            fuel_unit_str="kWh")

    assert_allclose(result['temp_sensitivity_params'], params, rtol=RTOL, atol=ATOL)
    assert result.get('n_days') is not None
    assert result.get('average_daily_usages') is not None
    assert result.get('estimated_average_daily_usages') is not None


@pytest.mark.slow
def test_annualized_usage_meter(generated_consumption_history_with_annualized_usage_1,
                               gsod_722880_2012_2014_weather_source,
                               tmy3_722880_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            sequence: [
                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter {
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
                            weather_normal_source=tmy3_722880_weather_source,
                            fuel_unit_str="kWh",
                            fuel_type="electricity")

    assert_allclose(result['temp_sensitivity_params'], params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['annualized_usage'], annualized_usage, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_gross_savings_metric(generated_consumption_history_pre_post_with_gross_savings_1,
                              gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            extras: {
                fuel_unit_str: "kWh",
                fuel_type: "electricity",
            },
            sequence: [
                !obj:eemeter.meter.PrePost {
                    splittable_args: ["consumption_history"],
                    pre_meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter &meter {
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
                    post_meter: *meter,
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
                            retrofit_completion_date=retrofit)

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
            extras: {
                fuel_unit_str: "kWh",
                fuel_type: "electricity",
            },
            sequence: [
                !obj:eemeter.meter.PrePost {
                    splittable_args: ["consumption_history"],
                    pre_meter: !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter &meter {
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
                    post_meter: *meter,
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
                            retrofit_completion_date=retrofit)

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

def test_for_each_fuel_type():
    meter_yaml = """
        !obj:eemeter.meter.ForEachFuelType {
            fuel_types: [electricity,natural_gas],
            fuel_unit_strs: [kWh,therms],
            meter: !obj:eemeter.meter.Sequence {
                sequence: [
                    !obj:eemeter.meter.DummyMeter {
                        input_mapping: {
                            fuel_type: value,
                        }
                    },
                    !obj:eemeter.meter.DummyMeter {
                        input_mapping: {
                            value_one: value,
                        },
                        output_mapping: {
                            result: result_one
                        }
                    }
                ]
            }
        }
    """
    meter = load(meter_yaml)

    result = meter.evaluate(value_one=1)

    assert result["result_electricity"] == "electricity"
    assert result["result_natural_gas"] == "natural_gas"
    assert result["result_one_electricity"] == 1
    assert result["result_one_natural_gas"] == 1

    with pytest.raises(ValueError):
        meter = load("!obj:eemeter.meter.ForEachFuelType { fuel_types:[electricity],fuel_unit_strs:[], meter: null }")

def test_time_span_meter(time_span_1):
    ch, fuel_type, n_days = time_span_1
    meter = TimeSpanMeter()
    assert n_days == meter.evaluate(consumption_history=ch,fuel_type=fuel_type)["time_span"]

def test_total_hdd_meter(generated_consumption_history_with_hdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, hdd, base, temp_unit = generated_consumption_history_with_hdd_1
    meter = TotalHDDMeter(base=base,temperature_unit_str=temp_unit)
    result = meter.evaluate(consumption_history=ch,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert_allclose(hdd,result["total_hdd"],rtol=RTOL,atol=ATOL)

def test_total_cdd_meter(generated_consumption_history_with_cdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, cdd, base, temp_unit = generated_consumption_history_with_cdd_1
    meter = TotalCDDMeter(base=base,temperature_unit_str=temp_unit)
    result = meter.evaluate(consumption_history=ch,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert_allclose(cdd,result["total_cdd"],rtol=RTOL,atol=ATOL)

def test_normal_annual_hdd(tmy3_722880_weather_source):
    meter = NormalAnnualHDD(base=65,temperature_unit_str="degF")
    result = meter.evaluate(weather_normal_source=tmy3_722880_weather_source)
    assert_allclose(result["normal_annual_hdd"],1578.588175669573,rtol=RTOL,atol=ATOL)

def test_normal_annual_cdd(tmy3_722880_weather_source):
    meter = NormalAnnualCDD(base=65,temperature_unit_str="degF")
    result = meter.evaluate(weather_normal_source=tmy3_722880_weather_source)
    assert_allclose(result["normal_annual_cdd"],1248.4575607999941,rtol=RTOL,atol=ATOL)

def test_n_periods_meeting_hdd_per_day_threshold(generated_consumption_history_with_n_periods_hdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, n_periods_lt, n_periods_gt, hdd = generated_consumption_history_with_n_periods_hdd_1
    meter_lt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="lt")
    meter_gt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="gt")
    result_lt = meter_lt.evaluate(consumption_history=ch,
                            hdd=hdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate(consumption_history=ch,
                            hdd=hdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert n_periods_lt == result_lt["n_periods"]
    assert n_periods_gt == result_gt["n_periods"]

def test_n_periods_meeting_cdd_per_day_threshold(generated_consumption_history_with_n_periods_cdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, n_periods_lt, n_periods_gt, cdd = generated_consumption_history_with_n_periods_cdd_1
    meter_lt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="lt")
    meter_gt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="gt")
    result_lt = meter_lt.evaluate(consumption_history=ch,
                            cdd=cdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate(consumption_history=ch,
                            cdd=cdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert n_periods_lt == result_lt["n_periods"]
    assert n_periods_gt == result_gt["n_periods"]

def test_recent_reading_meter():
    recent_consumption = Consumption(0,"kWh","electricity",datetime.now() - timedelta(days=390),datetime.now() - timedelta(days=360))
    old_consumption = Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1))
    no_ch = ConsumptionHistory([])
    old_ch = ConsumptionHistory([old_consumption])
    recent_ch = ConsumptionHistory([recent_consumption])
    mixed_ch = ConsumptionHistory([recent_consumption,old_consumption])

    meter = RecentReadingMeter(n_days=365)
    assert not meter.evaluate(consumption_history=no_ch,fuel_type="electricity")["recent_reading"]
    assert not meter.evaluate(consumption_history=old_ch,fuel_type="electricity")["recent_reading"]
    assert meter.evaluate(consumption_history=recent_ch,fuel_type="electricity")["recent_reading"]
    assert meter.evaluate(consumption_history=mixed_ch,fuel_type="electricity")["recent_reading"]
    assert not meter.evaluate(consumption_history=mixed_ch,fuel_type="natural_gas")["recent_reading"]

    assert not meter.evaluate(consumption_history=mixed_ch,fuel_type="electricity",
                              since_date=datetime.now() + timedelta(days=1000))["recent_reading"]
    assert not meter.evaluate(consumption_history=mixed_ch,fuel_type="natural_gas",
                              since_date=datetime.now() + timedelta(days=1000))["recent_reading"]

def test_cvrmse():
    meter = CVRMSE()
    result = meter.evaluate(y=np.array([12,13,414,12,23,12,32]),
                            y_hat=np.array([32,12,322,21,22,41,32]),
                            params=np.array([1,3,4]))

    assert_allclose(result["cvrmse"],66.84,rtol=RTOL,atol=ATOL)

def test_average_daily_usage(generated_consumption_history_1):
    ch,params = generated_consumption_history_1
    meter = AverageDailyUsage()
    result = meter.evaluate(consumption_history=ch,
                            fuel_type="electricity",
                            fuel_unit_str="kWh")
    assert result["average_daily_usages"] is not None

def test_estimated_average_daily_usage(generated_consumption_history_1,gsod_722880_2012_2014_weather_source):
    meter_yaml = """
        !obj:eemeter.meter.EstimatedAverageDailyUsage {
            temperature_unit_str: "degF",
            model: !obj:eemeter.models.TemperatureSensitivityModel {
                cooling: True,
                heating: True,
            },
        }
        """
    meter = load(meter_yaml)

    ch,params = generated_consumption_history_1

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            temp_sensitivity_params=params,
                            fuel_type="electricity",
                            fuel_unit_str="kWh")
    assert result["estimated_average_daily_usages"] is not None
    assert result["n_days"] is not None

def test_rmse():
    meter = RMSE()
    result = meter.evaluate(y=np.array([12,13,414,12,23,12,32]),
                            y_hat=np.array([32,12,322,21,22,41,32]))

    assert_allclose(result["rmse"],37.3898,rtol=RTOL,atol=ATOL)

def test_r_squared():
    meter = RSquared()
    result = meter.evaluate(y=np.array([12,13,414,12,23,12,32]),
                            y_hat=np.array([32,12,322,21,22,41,32]))

    assert_allclose(result["r_squared"],0.9276,rtol=RTOL,atol=ATOL)
