from eemeter.config.yaml_parser import load

from eemeter.consumption import ConsumptionData

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

from fixtures.consumption import consumption_generator_1
from fixtures.consumption import consumption_generator_2
from fixtures.consumption import generated_consumption_data_1
from fixtures.consumption import generated_consumption_data_2
from fixtures.consumption import generated_consumption_data_pre_post_1
from fixtures.consumption import generated_consumption_data_with_annualized_usage_1
from fixtures.consumption import generated_consumption_data_pre_post_with_gross_savings_1
from fixtures.consumption import generated_consumption_data_pre_post_with_annualized_gross_savings_1
from fixtures.consumption import consumption_data_1
from fixtures.consumption import time_span_1
from fixtures.consumption import generated_consumption_data_with_hdd_1
from fixtures.consumption import generated_consumption_data_with_cdd_1
from fixtures.consumption import generated_consumption_data_with_n_periods_hdd_1
from fixtures.consumption import generated_consumption_data_with_n_periods_cdd_1

from datetime import datetime
from datetime import timedelta
import pytz

from numpy.testing import assert_allclose
import numpy as np

RTOL = 1e-2
ATOL = 1e-2

import pytest

@pytest.mark.slow
def test_temperature_sensitivity_parameter_optimization(
        generated_consumption_data_1, gsod_722880_2012_2014_weather_source):

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

    cd, params = generated_consumption_data_1

    result = meter.evaluate(consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source,
            fuel_unit_str="kWh")

    assert_allclose(result['temp_sensitivity_params'], params, rtol=RTOL,
            atol=ATOL)
    assert result.get('n_days') is not None
    assert result.get('average_daily_usages') is not None
    assert result.get('estimated_average_daily_usages') is not None


@pytest.mark.slow
def test_annualized_usage_meter(
        generated_consumption_data_with_annualized_usage_1,
        gsod_722880_2012_2014_weather_source, tmy3_722880_weather_source):

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
                    output_mapping: {
                        temp_sensitivity_params: model_params,
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

    cd, params, annualized_usage = \
            generated_consumption_data_with_annualized_usage_1
    result = meter.evaluate(consumption_data=cd,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source,
                            fuel_unit_str="kWh")

    assert_allclose(result['model_params'], params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['annualized_usage'], annualized_usage, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_gross_savings_metric(generated_consumption_data_pre_post_with_gross_savings_1,
                              gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            extras: {
                fuel_unit_str: "kWh",
            },
            sequence: [
                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter &meter {
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
                    output_mapping: {
                        temp_sensitivity_params: model_params_baseline,
                    },
                },
                !obj:eemeter.meter.GrossSavingsMeter {
                    input_mapping: {
                        consumption_data: consumption_data_reporting,
                    },
                    fuel_unit_str: "kWh",
                    temperature_unit_str: "degF",
                    model: *model,
                }
            ]
        }
        """
    meter = load(meter_yaml)

    cd, _, _, retrofit, savings = \
            generated_consumption_data_pre_post_with_gross_savings_1

    result = meter.evaluate(consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source)

    assert_allclose(result["gross_savings"], savings, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_annualized_gross_savings_metric(
        generated_consumption_data_pre_post_with_annualized_gross_savings_1,
        gsod_722880_2012_2014_weather_source, tmy3_722880_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            extras: {
                fuel_unit_str: "kWh",
            },
            sequence: [
                !obj:eemeter.meter.TemperatureSensitivityParameterOptimizationMeter &meter {
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
                    output_mapping: {
                        temp_sensitivity_params: [
                            model_params_baseline,
                            model_params_reporting
                        ],
                    },
                },
                !obj:eemeter.meter.AnnualizedGrossSavingsMeter {
                    input_mapping: {
                        consumption_data: consumption_data_reporting,
                    },
                    temperature_unit_str: "degF",
                    model: *model,
                }
            ]
        }
        """
    meter = load(meter_yaml)

    cd, _, _, retrofit, savings = \
            generated_consumption_data_pre_post_with_annualized_gross_savings_1

    result = meter.evaluate(consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source,
            weather_normal_source=tmy3_722880_weather_source,
            retrofit_start_date=retrofit,
            retrofit_completion_date=retrofit)

    assert_allclose(result["annualized_gross_savings"], savings, rtol=RTOL, atol=ATOL)

def test_time_span_meter(time_span_1):
    cd, n_days = time_span_1
    meter = TimeSpanMeter()
    assert n_days == meter.evaluate(consumption_data=cd)["time_span"]

def test_total_hdd_meter(generated_consumption_data_with_hdd_1,gsod_722880_2012_2014_weather_source):
    cd, hdd, base, temp_unit = generated_consumption_data_with_hdd_1
    meter = TotalHDDMeter(base=base,temperature_unit_str=temp_unit)
    result = meter.evaluate(consumption_data=cd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert_allclose(hdd,result["total_hdd"],rtol=RTOL,atol=ATOL)

def test_total_cdd_meter(generated_consumption_data_with_cdd_1,gsod_722880_2012_2014_weather_source):
    cd, cdd, base, temp_unit = generated_consumption_data_with_cdd_1
    meter = TotalCDDMeter(base=base,temperature_unit_str=temp_unit)
    result = meter.evaluate(consumption_data=cd,
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

def test_n_periods_meeting_hdd_per_day_threshold(generated_consumption_data_with_n_periods_hdd_1,gsod_722880_2012_2014_weather_source):
    cd, n_periods_lt, n_periods_gt, hdd = generated_consumption_data_with_n_periods_hdd_1
    meter_lt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="lt")
    meter_gt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="gt")
    result_lt = meter_lt.evaluate(consumption_data=cd,
                            hdd=hdd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate(consumption_data=cd,
                            hdd=hdd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert n_periods_lt == result_lt["n_periods"]
    assert n_periods_gt == result_gt["n_periods"]

def test_n_periods_meeting_cdd_per_day_threshold(generated_consumption_data_with_n_periods_cdd_1,gsod_722880_2012_2014_weather_source):
    cd, n_periods_lt, n_periods_gt, cdd = generated_consumption_data_with_n_periods_cdd_1
    meter_lt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="lt")
    meter_gt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="gt")
    result_lt = meter_lt.evaluate(consumption_data=cd,
                            cdd=cdd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate(consumption_data=cd,
                            cdd=cdd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert n_periods_lt == result_lt["n_periods"]
    assert n_periods_gt == result_gt["n_periods"]

def test_recent_reading_meter():
    recent_record = {"start": datetime.now(pytz.utc) - timedelta(days=390),
            "end": datetime.now(pytz.utc) - timedelta(days=360), "value": 0}
    old_record = {"start": datetime(2012,1,1,tzinfo=pytz.utc),
            "end": datetime(2012,2,1,tzinfo=pytz.utc), "value": 0}

    no_cd = ConsumptionData([],
            "electricity", "kWh", record_type="arbitrary")
    old_cd = ConsumptionData([old_record],
            "electricity", "kWh", record_type="arbitrary")
    recent_cd = ConsumptionData([recent_record],
            "electricity", "kWh", record_type="arbitrary")
    mixed_cd = ConsumptionData([recent_record,old_record],
            "electricity", "kWh", record_type="arbitrary")

    meter = RecentReadingMeter(n_days=365)
    assert not meter.evaluate(consumption_data=no_cd)["recent_reading"]
    assert not meter.evaluate(consumption_data=old_cd)["recent_reading"]
    assert meter.evaluate(consumption_data=recent_cd)["recent_reading"]
    assert meter.evaluate(consumption_data=mixed_cd)["recent_reading"]

    since_date = datetime.now(pytz.utc) + timedelta(days=1000)
    assert not meter.evaluate(consumption_data=mixed_cd,
            since_date=since_date)["recent_reading"]

def test_cvrmse():
    meter = CVRMSE()
    result = meter.evaluate(y=np.array([12,13,414,12,23,12,32,np.nan]),
                            y_hat=np.array([32,12,322,21,22,41,32,np.nan]),
                            params=np.array([1,3,4]))

    assert_allclose(result["cvrmse"],59.79,rtol=RTOL,atol=ATOL)

def test_average_daily_usage(generated_consumption_data_1):
    cd,params = generated_consumption_data_1
    meter = AverageDailyUsage()
    result = meter.evaluate(consumption_data=cd,
                            fuel_unit_str="kWh")
    assert result["average_daily_usages"] is not None

def test_estimated_average_daily_usage(generated_consumption_data_1,gsod_722880_2012_2014_weather_source):
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

    cd,params = generated_consumption_data_1

    result = meter.evaluate(consumption_data=cd,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            temp_sensitivity_params=params,
                            fuel_unit_str="kWh")
    assert result["estimated_average_daily_usages"] is not None
    assert result["n_days"] is not None

def test_rmse():
    meter = RMSE()
    result = meter.evaluate(y=np.array([12,13,414,12,23,12,32,np.nan]),
                            y_hat=np.array([32,12,322,21,22,41,32,np.nan]))

    assert_allclose(result["rmse"],34.97,rtol=RTOL,atol=ATOL)

def test_r_squared():
    meter = RSquared()
    result = meter.evaluate(y=np.array([12,13,414,12,23,12,32,np.nan]),
                            y_hat=np.array([32,12,322,21,22,41,32,np.nan]))

    assert_allclose(result["r_squared"],0.9276,rtol=RTOL,atol=ATOL)
