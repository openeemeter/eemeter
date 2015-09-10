from eemeter.meter import DefaultResidentialMeter
from eemeter.meter import DataCollection
from eemeter.models import AverageDailyTemperatureSensitivityModel
from eemeter.generator import MonthlyBillingConsumptionGenerator
from eemeter.generator import generate_monthly_billing_datetimes
from eemeter.consumption import ConsumptionData
from eemeter.evaluation import Period
from eemeter.location import Location
from eemeter.project import Project

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from numpy.testing import assert_allclose
import numpy as np
from scipy.stats import randint

from datetime import datetime
import pytz

RTOL = 1e-2
ATOL = 1e-2

import pytest

@pytest.fixture(params=[([1, 14.5,1,17.8,8], [1, 14.5,1], 6849.29, 615.69,
                            0, 0, 1, 1, "degC", 693.58, 877.00, 2587.61, 1805.00),
                        ([3,15.5,2,19.5,1], [3,15.5,2], 2374.35, 1848.67,
                            0, 0, 1, 1, "degC", 693.58, 877.00,2587.61, 1805.00),
                        ([0,18.8,2,22.2,7], [0,18.8,2], 3616.24, 1954.77,
                            0, 0, 1, 1, "degC", 693.58, 877.00, 2587.61, 1805.00),
                        ([0,65,2,71,3], [0,65,2], 4700.22, 3157.17,
                             0, 0, 1, 1, "degF", 1248.45, 1578.58, 4657.70, 3249.00),
                        ])
def default_residential_outputs_1(request, gsod_722880_2012_2014_weather_source):
    elec_model_params, gas_model_params, \
            elec_annualized_usage, gas_annualized_usage, \
            elec_rmse, gas_rmse, elec_r_squared, gas_r_squared, \
            temp_unit, cdd_tmy, hdd_tmy, total_cdd, total_hdd \
            = request.param

    period = Period(datetime(2012,1,1,tzinfo=pytz.utc),
            datetime(2014,12,31,tzinfo=pytz.utc))
    retrofit_start_date = datetime(2013,6,1,tzinfo=pytz.utc)
    retrofit_completion_date = datetime(2013,8,1,tzinfo=pytz.utc)

    datetimes = generate_monthly_billing_datetimes(period,randint(30,31))

    # generate electricity consumption
    elec_model = AverageDailyTemperatureSensitivityModel(cooling=True,heating=True)
    elec_params = {
        "base_daily_consumption": elec_model_params[0],
        "heating_balance_temperature": elec_model_params[1],
        "heating_slope": elec_model_params[2],
        "cooling_balance_temperature": elec_model_params[3],
        "cooling_slope": elec_model_params[4],
    }
    elec_gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", temp_unit,
            elec_model, elec_params)
    elec_consumption_data = elec_gen.generate(gsod_722880_2012_2014_weather_source, datetimes)
    elec_consumption_kWh_per_day, elec_consumption_n_days = \
            elec_consumption_data.average_daily_consumptions()
    elec_params = elec_model.param_type(elec_params)

    # generate natural_gas consumption
    gas_model = AverageDailyTemperatureSensitivityModel(cooling=False,heating=True)
    gas_params = {
        "base_daily_consumption": gas_model_params[0],
        "heating_balance_temperature": gas_model_params[1],
        "heating_slope": gas_model_params[2],
    }
    gas_gen = MonthlyBillingConsumptionGenerator("natural_gas", "therm", temp_unit,
            gas_model, gas_params)
    gas_consumption_data = gas_gen.generate(gsod_722880_2012_2014_weather_source, datetimes)
    gas_consumption_kWh_per_day, gas_consumption_n_days = \
            gas_consumption_data.average_daily_consumptions()
    gas_params = gas_model.param_type(gas_params)

    fixture = elec_consumption_data, gas_consumption_data, \
            elec_params, gas_params, \
            elec_annualized_usage, gas_annualized_usage, \
            elec_rmse, gas_rmse, \
            elec_r_squared, gas_r_squared, \
            elec_consumption_kWh_per_day, gas_consumption_kWh_per_day, \
            elec_consumption_n_days, gas_consumption_n_days, \
            temp_unit, retrofit_start_date, retrofit_completion_date, \
            cdd_tmy, hdd_tmy, total_cdd, total_hdd
    return fixture

@pytest.mark.slow
def test_default_residential_meter(default_residential_outputs_1,
        gsod_722880_2012_2014_weather_source, tmy3_722880_weather_source):
    elec_consumption_data, gas_consumption_data, \
            elec_params, gas_params, \
            elec_annualized_usage, gas_annualized_usage, \
            elec_rmse, gas_rmse, \
            elec_r_squared, gas_r_squared, \
            elec_consumption_kWh_per_day, gas_consumption_kWh_per_day, \
            elec_consumption_n_days, gas_consumption_n_days, \
            temp_unit, retrofit_start_date, retrofit_completion_date, \
            cdd_tmy, hdd_tmy, total_cdd, total_hdd = \
            default_residential_outputs_1

    meter = DefaultResidentialMeter(temperature_unit_str=temp_unit)

    location = Location(station="722880")
    baseline_period = Period(datetime(2012,1,1,tzinfo=pytz.utc),retrofit_start_date)
    reporting_period = Period(retrofit_completion_date, datetime(2014,12,31,tzinfo=pytz.utc))
    project = Project(location, [elec_consumption_data, gas_consumption_data], baseline_period,
            reporting_period)

    data_collection = DataCollection(project=project)
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data('annualized_usage',
            tags=['electricity','baseline']).value, elec_annualized_usage,
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('annualized_usage',
            tags=['natural_gas','baseline']).value, gas_annualized_usage,
            rtol=RTOL, atol=ATOL)


    assert_allclose(result.get_data('annualized_usage',
            tags=['electricity','reporting']).value, elec_annualized_usage,
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('annualized_usage',
            tags=['natural_gas','reporting']).value, gas_annualized_usage,
            rtol=RTOL, atol=ATOL)


    assert_allclose(result.get_data('average_daily_usages',
            tags=['electricity', 'baseline']).value, elec_consumption_kWh_per_day[:17],
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('average_daily_usages',
            tags=['natural_gas', 'baseline']).value, gas_consumption_kWh_per_day[:17],
            rtol=RTOL, atol=ATOL)


    assert_allclose(result.get_data('average_daily_usages',
            tags=['electricity', 'reporting']).value, elec_consumption_kWh_per_day[20:],
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('average_daily_usages',
            tags=['natural_gas', 'reporting']).value, gas_consumption_kWh_per_day[20:],
            rtol=RTOL, atol=ATOL)


    assert_allclose(result.get_data('cdd_tmy',
            tags=['electricity', 'baseline']).value, cdd_tmy,
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('cdd_tmy',
            tags=['natural_gas', 'baseline']).value, cdd_tmy,
            rtol=RTOL, atol=ATOL)


    assert_allclose(result.get_data('cdd_tmy',
            tags=['electricity', 'reporting']).value, cdd_tmy,
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('cdd_tmy',
            tags=['natural_gas', 'reporting']).value, cdd_tmy,
            rtol=RTOL, atol=ATOL)


    elec_rmse_results = result.search('rmse', ['electricity'])
    assert elec_rmse_results.count() == 2
    for data_container in elec_rmse_results.iteritems():
        assert_allclose(data_container.value, elec_rmse, rtol=RTOL, atol=ATOL)

    gas_rmse_results = result.search('rmse', ['natural_gas'])
    assert gas_rmse_results.count() == 2
    for data_container in gas_rmse_results.iteritems():
        assert_allclose(data_container.value, gas_rmse, rtol=RTOL, atol=ATOL)

    elec_r_squared_results = result.search('r_squared', ['electricity'])
    assert elec_r_squared_results.count() == 2
    for data_container in elec_r_squared_results.iteritems():
        assert_allclose(data_container.value, elec_r_squared, rtol=RTOL, atol=ATOL)

    gas_r_squared_results = result.search('r_squared', ['natural_gas'])
    assert gas_r_squared_results.count() == 2
    for data_container in gas_r_squared_results.iteritems():
        assert_allclose(data_container.value, gas_r_squared, rtol=RTOL, atol=ATOL)


def test_default_residential_meter_bad_inputs():

    with pytest.raises(ValueError):
        DefaultResidentialMeter(temperature_unit_str="unexpected")

    with pytest.raises(ValueError):
        DefaultResidentialMeter(settings={"heating_balance_temp_high":0,"heating_balance_temp_x0":1,"heating_balance_temp_low":2})

    with pytest.raises(ValueError):
        DefaultResidentialMeter(settings={"cooling_balance_temp_high":0,"cooling_balance_temp_x0":1,"cooling_balance_temp_low":2})

    with pytest.raises(ValueError):
        DefaultResidentialMeter(settings={"electricity_heating_slope_high":-1})

    with pytest.raises(ValueError):
        DefaultResidentialMeter(settings={"natural_gas_heating_slope_high":-1})

    with pytest.raises(ValueError):
        DefaultResidentialMeter(settings={"electricity_cooling_slope_high":-1})

