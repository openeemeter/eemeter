from eemeter.config.yaml_parser import load
from eemeter.meter import DataCollection

from eemeter.consumption import ConsumptionData
from eemeter.evaluation import Period
from eemeter.location import Location
from eemeter.project import Project

from eemeter.meter import TimeSpanMeter
from eemeter.meter import TotalHDDMeter
from eemeter.meter import TotalCDDMeter
from eemeter.meter import NormalAnnualHDD
from eemeter.meter import NormalAnnualCDD
from eemeter.meter import NPeriodsMeetingHDDPerDayThreshold
from eemeter.meter import NPeriodsMeetingCDDPerDayThreshold
from eemeter.meter import RecentReadingMeter
from eemeter.meter import AverageDailyUsage
from eemeter.meter import EstimatedAverageDailyUsage
from eemeter.meter import ConsumptionDataAttributes
from eemeter.meter import ProjectAttributes
from eemeter.meter import ProjectConsumptionDataBaselineReporting

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
            input_mapping: {
                "consumption_data": {},
                "weather_source": {},
                "energy_unit_str": {},
            },
            output_mapping: {
                "temp_sensitivity_params": {},
                "n_days": {},
                "average_daily_usages": {},
                "estimated_average_daily_usages": {},
            },
        }
        """
    meter = load(meter_yaml)

    cd, params = generated_consumption_data_1

    data_collection = DataCollection(
            consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source,
            energy_unit_str="kWh")

    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data('temp_sensitivity_params').value, params,
            rtol=RTOL, atol=ATOL)
    assert result.get_data('n_days') is not None
    assert result.get_data('average_daily_usages') is not None
    assert result.get_data('estimated_average_daily_usages') is not None


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
                    input_mapping: {
                        consumption_data: {},
                        weather_source: {},
                        energy_unit_str: {},
                    },
                    output_mapping: {
                        temp_sensitivity_params: {name: model_params},
                    },
                },
                !obj:eemeter.meter.AnnualizedUsageMeter {
                    temperature_unit_str: "degF",
                    model: *model,
                    input_mapping: {
                        model_params: {},
                        weather_normal_source: {},
                    },
                    output_mapping: {
                        annualized_usage: {},
                    },
                }
            ]
        }
        """
    meter = load(meter_yaml)

    cd, params, annualized_usage = \
            generated_consumption_data_with_annualized_usage_1

    data_collection = DataCollection(
            consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source,
            weather_normal_source=tmy3_722880_weather_source,
            energy_unit_str="kWh")
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data('model_params').value, params,
            rtol=RTOL, atol=ATOL)
    assert_allclose(result.get_data('annualized_usage').value,
            annualized_usage, rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_gross_savings_metric(generated_consumption_data_pre_post_with_gross_savings_1,
                              gsod_722880_2012_2014_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
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
                    input_mapping: {
                        consumption_data: {},
                        weather_source: {},
                        energy_unit_str: {},
                    },
                    output_mapping: {
                        temp_sensitivity_params: {name: model_params},
                    },
                },
                !obj:eemeter.meter.GrossSavingsMeter {
                    temperature_unit_str: "degF",
                    model: *model,
                    input_mapping: {
                        model_params_baseline: {name: model_params},
                        consumption_data_reporting: {name: consumption_data},
                        weather_source: {},
                        energy_unit_str: {},
                    },
                    output_mapping: {
                        gross_savings: {},
                    },
                }
            ]
        }
        """
    meter = load(meter_yaml)

    cd, _, _, retrofit, savings = \
            generated_consumption_data_pre_post_with_gross_savings_1

    data_collection = DataCollection(
            consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source,
            weather_normal_source=tmy3_722880_weather_source,
            energy_unit_str="kWh")
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data("gross_savings").value, savings,
            rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_annualized_gross_savings_metric(
        generated_consumption_data_pre_post_with_annualized_gross_savings_1,
        gsod_722880_2012_2014_weather_source, tmy3_722880_weather_source):

    meter_yaml = """
        !obj:eemeter.meter.Sequence {
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
                    input_mapping: {
                        consumption_data: {},
                        weather_source: {},
                        energy_unit_str: {},
                    },
                    output_mapping: {
                        temp_sensitivity_params: {},
                    },
                },
                !obj:eemeter.meter.AnnualizedGrossSavingsMeter {
                    temperature_unit_str: "degF",
                    model: *model,
                    input_mapping: {
                        model_params_baseline: {name: temp_sensitivity_params},
                        model_params_reporting: {name: temp_sensitivity_params},
                        consumption_data_reporting: {name: consumption_data},
                        weather_normal_source: {},
                    },
                    output_mapping: {
                        annualized_gross_savings: {},
                    },
                }
            ]
        }
        """
    meter = load(meter_yaml)

    cd, _, _, retrofit, savings = \
            generated_consumption_data_pre_post_with_annualized_gross_savings_1

    data_collection = DataCollection(
            consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source,
            weather_normal_source=tmy3_722880_weather_source,
            retrofit_start_date=retrofit,
            retrofit_completion_date=retrofit,
            energy_unit_str="kWh")
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data("annualized_gross_savings").value, savings,
            rtol=RTOL, atol=ATOL)

def test_time_span_meter(time_span_1):
    cd, n_days = time_span_1
    meter = TimeSpanMeter()
    assert n_days == meter.evaluate_raw(consumption_data=cd)["time_span"]

def test_total_hdd_meter(generated_consumption_data_with_hdd_1,gsod_722880_2012_2014_weather_source):
    cd, hdd, base, temp_unit = generated_consumption_data_with_hdd_1
    meter = TotalHDDMeter(base=base,temperature_unit_str=temp_unit)
    result = meter.evaluate_raw(consumption_data=cd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert_allclose(hdd,result["total_hdd"],rtol=RTOL,atol=ATOL)

def test_total_cdd_meter(generated_consumption_data_with_cdd_1,gsod_722880_2012_2014_weather_source):
    cd, cdd, base, temp_unit = generated_consumption_data_with_cdd_1
    meter = TotalCDDMeter(base=base,temperature_unit_str=temp_unit)
    result = meter.evaluate_raw(consumption_data=cd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert_allclose(cdd,result["total_cdd"],rtol=RTOL,atol=ATOL)

def test_normal_annual_hdd(tmy3_722880_weather_source):
    meter = NormalAnnualHDD(base=65,temperature_unit_str="degF")
    result = meter.evaluate_raw(weather_normal_source=tmy3_722880_weather_source)
    assert_allclose(result["normal_annual_hdd"],1578.588175669573,rtol=RTOL,atol=ATOL)

def test_normal_annual_cdd(tmy3_722880_weather_source):
    meter = NormalAnnualCDD(base=65,temperature_unit_str="degF")
    result = meter.evaluate_raw(weather_normal_source=tmy3_722880_weather_source)
    assert_allclose(result["normal_annual_cdd"],1248.4575607999941,rtol=RTOL,atol=ATOL)

def test_n_periods_meeting_hdd_per_day_threshold(generated_consumption_data_with_n_periods_hdd_1,gsod_722880_2012_2014_weather_source):
    cd, n_periods_lt, n_periods_gt, hdd = generated_consumption_data_with_n_periods_hdd_1
    meter_lt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="<")
    meter_gt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation=">")
    result_lt = meter_lt.evaluate_raw(consumption_data=cd,
                            hdd=hdd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate_raw(consumption_data=cd,
                            hdd=hdd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert n_periods_lt == result_lt["n_periods"]
    assert n_periods_gt == result_gt["n_periods"]

def test_n_periods_meeting_cdd_per_day_threshold(generated_consumption_data_with_n_periods_cdd_1,gsod_722880_2012_2014_weather_source):
    cd, n_periods_lt, n_periods_gt, cdd = generated_consumption_data_with_n_periods_cdd_1
    meter_lt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="<")
    meter_gt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation=">")
    result_lt = meter_lt.evaluate_raw(consumption_data=cd,
                            cdd=cdd,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate_raw(consumption_data=cd,
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

    meter = RecentReadingMeter()
    assert meter.evaluate_raw(consumption_data=no_cd)["n_days"] == np.inf
    assert meter.evaluate_raw(consumption_data=old_cd)["n_days"] == 31
    assert meter.evaluate_raw(consumption_data=recent_cd)["n_days"] == 30
    assert meter.evaluate_raw(consumption_data=mixed_cd)["n_days"] == 30

def test_average_daily_usage(generated_consumption_data_1):
    cd,params = generated_consumption_data_1
    meter = AverageDailyUsage()
    result = meter.evaluate_raw(consumption_data=cd,
                            energy_unit_str="kWh")
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

    result = meter.evaluate_raw(
            consumption_data=cd,
            weather_source=gsod_722880_2012_2014_weather_source,
            temp_sensitivity_params=params,
            energy_unit_str="kWh")
    assert result["estimated_average_daily_usages"] is not None
    assert result["n_days"] is not None

def test_consumption_data_attributes(generated_consumption_data_1):
    cd,params = generated_consumption_data_1
    meter = ConsumptionDataAttributes()
    result = meter.evaluate_raw(consumption_data=cd)
    assert result["fuel_type"] == "electricity"
    assert result["unit_name"] == "kWh"
    assert result["freq"] == None
    assert result["freq_timedelta"] == None
    assert result["pulse_value"] == None
    assert result["name"] == None

def test_project_attributes(generated_consumption_data_1):
    cd,params = generated_consumption_data_1
    baseline_period = Period(datetime(2014,1,1),datetime(2014,1,1))
    location = Location(zipcode="91104")
    project = Project(location,[cd],baseline_period,None)
    meter = ProjectAttributes(project)
    result = meter.evaluate_raw(project=project)
    assert result["location"].zipcode == location.zipcode
    assert result["consumption"][0] is not None
    assert result["baseline_period"] is not None
    assert result["reporting_period"] is None
    assert result["other_periods"] == []
    assert result["weather_source"].station_id == "722880"
    assert result["weather_normal_source"].station_id == "722880"

def test_project_consumption_baseline_reporting(generated_consumption_data_1):
    cd, _ = generated_consumption_data_1
    baseline_period = Period(datetime(2011,1,1),datetime(2013,6,1))
    reporting_period = Period(datetime(2013,6,1),datetime(2016,1,1))
    location = Location(zipcode="91104")
    project = Project(location,[cd],baseline_period,reporting_period)
    meter = ProjectConsumptionDataBaselineReporting()
    result = meter.evaluate_raw(project=project)
    assert result["consumption"][0]["value"].data.index[0] == datetime(2012,1,1)
    assert result["consumption"][0]["value"].data.index[17] == datetime(2013,5,25)
    assert result["consumption"][0]["tags"][0] == "electricity"
    assert result["consumption"][0]["tags"][1] == "baseline"
    assert result["consumption"][1]["value"].data.index[0] == datetime(2013,6,24)
    assert result["consumption"][1]["value"].data.index[18] == datetime(2014,12,16)
    assert result["consumption"][1]["tags"][0] == "electricity"
    assert result["consumption"][1]["tags"][1] == "reporting"
