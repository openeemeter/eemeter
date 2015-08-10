from eemeter.generator import MonthlyBillingConsumptionGenerator
from eemeter.generator import ProjectGenerator
from eemeter.generator import generate_monthly_billing_datetimes

from eemeter.evaluation import Period
from eemeter.models.temperature_sensitivity import TemperatureSensitivityModel
from eemeter.location import Location

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

import pytest
from datetime import datetime
from datetime import timedelta

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import uniform
from scipy.stats import randint

RTOL=1e-2
ATOL=1e-2

@pytest.fixture
def monthly_datetimes_2012():
    datetimes = [datetime(2012,i,1) for i in range(1,13)]
    datetimes.append(datetime(2013,1,1))
    return datetimes

@pytest.fixture
def consumption_generator_no_base_load():
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": 0.0,
        "heating_slope": 1.0,
        "heating_reference_temperature": 65.0,
        "cooling_slope": 1.0,
        "cooling_reference_temperature": 75.0 }
    generator = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, params)
    return generator

@pytest.fixture
def consumption_generator_with_base_load():
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": 1.0,
        "heating_slope": 1.0,
        "heating_reference_temperature": 65.0,
        "cooling_slope": 1.0,
        "cooling_reference_temperature": 75.0 }
    generator = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, params)
    return generator

@pytest.mark.slow
def test_consumption_generator_no_base_load(consumption_generator_no_base_load,
        gsod_722880_2012_2014_weather_source, monthly_datetimes_2012):
    consumption_data = consumption_generator_no_base_load.generate(
            gsod_722880_2012_2014_weather_source, monthly_datetimes_2012)
    assert_allclose(consumption_data.data.values,
            [241.5, 279.2, 287.6, 139.2, 56.2, 2.1,
             22.6, 154.3, 106.8, 53.4, 137.9, 351.1, np.nan],
            rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_consumption_generator_with_base_load(
        consumption_generator_with_base_load,
        gsod_722880_2012_2014_weather_source, monthly_datetimes_2012):
    consumption_data = consumption_generator_with_base_load.generate(
            gsod_722880_2012_2014_weather_source, monthly_datetimes_2012)
    assert_allclose(consumption_data.data.values,
            [272.5, 308.2, 318.6, 169.2, 87.2, 32.1,
             53.6, 185.3, 136.8, 84.4, 167.9, 382.1, np.nan],
            rtol=RTOL, atol=ATOL)

@pytest.mark.slow
def test_project_generator(gsod_722880_2012_2014_weather_source,tmy3_722880_weather_source):
    electricity_model = TemperatureSensitivityModel(cooling=True,heating=True)
    gas_model = TemperatureSensitivityModel(cooling=False,heating=True)
    electricity_param_distributions = {
            "cooling_slope": uniform(loc=1, scale=.5),
            "heating_slope": uniform(loc=1, scale=.5),
            "base_consumption": uniform(loc=5, scale=5),
            "cooling_reference_temperature": uniform(loc=70, scale=5),
            "heating_reference_temperature": uniform(loc=60, scale=5)}
    electricity_param_delta_distributions = {
            "cooling_slope": uniform(loc=-.2, scale=.3),
            "heating_slope": uniform(loc=-.2, scale=.3),
            "base_consumption": uniform(loc=-2, scale=3),
            "cooling_reference_temperature": uniform(loc=0, scale=0),
            "heating_reference_temperature": uniform(loc=0, scale=0)}
    gas_param_distributions = {
            "heating_slope": uniform(loc=1, scale=.5),
            "base_consumption": uniform(loc=5, scale=5),
            "heating_reference_temperature": uniform(loc=60, scale=5)}
    gas_param_delta_distributions = {
            "heating_slope": uniform(loc=-.2, scale=.3),
            "base_consumption": uniform(loc=-2, scale=3),
            "heating_reference_temperature": uniform(loc=0, scale=0)}

    generator = ProjectGenerator(electricity_model, gas_model,
                                electricity_param_distributions,
                                electricity_param_delta_distributions,
                                gas_param_distributions,
                                gas_param_delta_distributions)

    location = Location(station="722880")
    period = Period(datetime(2012,1,1),datetime(2013,1,1))
    baseline_period = Period(datetime(2012,1,1),datetime(2012,4,1))
    reporting_period = Period(datetime(2012,5,1),datetime(2013,1,1))

    results = generator.generate(location, period, period,
                               baseline_period, reporting_period)

    project = results["project"]
    elec_data = project.consumption[0].data.values
    gas_data = project.consumption[1].data.values
    assert project.location == location
    assert results.get("electricity_estimated_savings") is not None
    assert results.get("natural_gas_estimated_savings") is not None
    assert results.get("electricity_pre_params") is not None
    assert results.get("natural_gas_pre_params") is not None
    assert results.get("electricity_post_params") is not None
    assert results.get("natural_gas_post_params") is not None

    assert len(elec_data) in range(9,16)
    assert len(gas_data) in range(9,16)
    assert elec_data[0] < 750 # could probably lower this upper bound
    assert gas_data[0] < 750 # could probably lower this upper bound

def test_generate_monthly_billing_datetimes():
    period = Period(datetime(2012,1,1),datetime(2013,1,1))
    datetimes_30d = generate_monthly_billing_datetimes(period,
            randint(30,31))
    assert datetimes_30d[0] == datetime(2012,1,1)
    assert datetimes_30d[1] == datetime(2012,1,31)
    assert datetimes_30d[11] == datetime(2012,11,26)
    assert datetimes_30d[12] == datetime(2012,12,26)

    datetimes_1d = generate_monthly_billing_datetimes(period, randint(1,2))
    assert datetimes_1d[0] == datetime(2012,1,1)
    assert datetimes_1d[1] == datetime(2012,1,2)
    assert datetimes_1d[330] == datetime(2012,11,26)
    assert datetimes_1d[331] == datetime(2012,11,27)
