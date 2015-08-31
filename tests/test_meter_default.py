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

@pytest.fixture(params=[([-1, 14.5,1,17.8,8],True,6119.297438069778,0,"degC",
                          693.5875,876.9934,2587.6109,1805.0001,0,1,[0,0,15.55]),
                        ([10,15.5,2,19.5,1],True,4927.478974253085,0,"degC",
                          693.5875,876.9934,2587.6109,1805.0001,0,1,[0,0,15.55]),
                        ([0,18.8,2,22.2,7],True,3616.249477948155,0,"degC",
                          693.5875,876.9934,2587.6109,1805.0001,0,1,[0,0,15.55]),
                        ([0,65,2,71,3],True,4700.226534599519,0,"degF",
                          1248.4575,1578.5882,4657.6997,3249.0002,0,1,[0,0,60]),
                        ])
def default_residential_outputs_1(request, gsod_722880_2012_2014_weather_source):
    model_params, elec_presence, elec_annualized_usage, elec_error, \
            temp_unit, cdd_tmy, hdd_tmy, total_cdd, total_hdd, \
            rmse_electricity, r_squared_electricity, gas_param_defaults \
            = request.param
    model = AverageDailyTemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_daily_consumption": model_params[0],
        "heating_balance_temperature": model_params[1],
        "heating_slope": model_params[2],
        "cooling_balance_temperature": model_params[3],
        "cooling_slope": model_params[4],
    }

    period = Period(datetime(2012,1,1,tzinfo=pytz.utc),
            datetime(2014,12,31,tzinfo=pytz.utc))
    retrofit_start_date = datetime(2013,6,1,tzinfo=pytz.utc)
    retrofit_completion_date = datetime(2013,8,1,tzinfo=pytz.utc)

    datetimes = generate_monthly_billing_datetimes(period,randint(30,31))
    gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", temp_unit,
            model, params)
    consumption_data = gen.generate(gsod_722880_2012_2014_weather_source, datetimes)
    consumption_kWh_per_day, consumption_n_days = \
            consumption_data.average_daily_consumptions()

    elec_params = model.param_type(params)

    fixture = consumption_data, elec_params, elec_presence, \
            elec_annualized_usage, elec_error, temp_unit, \
            retrofit_start_date, retrofit_completion_date, \
            cdd_tmy, hdd_tmy, total_cdd, total_hdd, \
            rmse_electricity, r_squared_electricity, \
            consumption_kWh_per_day, consumption_kWh_per_day, \
            consumption_n_days, gas_param_defaults
    return fixture

@pytest.mark.slow
def test_default_residential_meter(default_residential_outputs_1,
        gsod_722880_2012_2014_weather_source, tmy3_722880_weather_source):
    consumption_data, elec_params, elec_presence, \
            elec_annualized_usage, elec_error, temp_unit, \
            retrofit_start_date, retrofit_completion_date, \
            cdd_tmy, hdd_tmy, total_cdd, total_hdd, \
            rmse_electricity, r_squared_electricity, \
            average_daily_usages, estimated_average_daily_usages, \
            n_days, gas_param_defaults = default_residential_outputs_1

    meter = DefaultResidentialMeter(temperature_unit_str=temp_unit)

    location = Location(station="722880")
    baseline_period = Period(datetime(2012,1,1,tzinfo=pytz.utc),retrofit_start_date)
    reporting_period = Period(retrofit_completion_date, datetime(2014,12,31,tzinfo=pytz.utc))
    project = Project(location, [consumption_data], baseline_period,
            reporting_period)

    data_collection = DataCollection(project=project)
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data('annualized_usage',
            tags=['electricity','baseline']).value, elec_annualized_usage,
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('annualized_usage',
            tags=['electricity','reporting']).value, elec_annualized_usage,
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('average_daily_usages',
            tags=['electricity', 'baseline']).value, average_daily_usages[:17],
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('average_daily_usages',
            tags=['electricity', 'reporting']).value, average_daily_usages[20:],
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('cdd_tmy',
            tags=['electricity', 'baseline']).value, cdd_tmy,
            rtol=RTOL, atol=ATOL)

    assert_allclose(result.get_data('cdd_tmy',
            tags=['electricity', 'reporting']).value, cdd_tmy,
            rtol=RTOL, atol=ATOL)

    assert result.get_data('cvrmse',
            tags=['electricity', 'baseline']).value < 2

    assert result.get_data('cvrmse',
            tags=['electricity', 'reporting']).value < 2


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

