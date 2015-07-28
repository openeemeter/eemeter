from eemeter.meter import PRISMMeter
from eemeter.models import TemperatureSensitivityModel
from eemeter.generator import MonthlyBillingConsumptionGenerator
from eemeter.generator import generate_monthly_billing_datetimes
from eemeter.consumption import ConsumptionData

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from numpy.testing import assert_allclose
import numpy as np

from datetime import datetime
import pytz

RTOL = 1e-2
ATOL = 1e-2

import pytest

@pytest.fixture(params=[([-1, 1,14.5,8,17.8],True,6119.297438069778,0,"degC",
                          693.5875,876.9934,2587.6109,1805.0001,0,1,[0,0,15.55]),
                        ([10,2,15.5,1,19.5],True,4927.478974253085,0,"degC",
                          693.5875,876.9934,2587.6109,1805.0001,0,1,[0,0,15.55]),
                        ([0,2,18.8,7,22.2],True,3616.249477948155,0,"degC",
                          693.5875,876.9934,2587.6109,1805.0001,0,1,[0,0,15.55]),
                        ([0,2,65,3,71],True,4700.226534599519,0,"degF",
                          1248.4575,1578.5882,4657.6997,3249.0002,0,1,[0,0,60]),
                        ])
def prism_outputs_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    start = datetime(2012,1,1,tzinfo=pytz.utc)
    end = datetime(2014,12,31,tzinfo=pytz.utc)
    retrofit_start_date = datetime(2013,6,1,tzinfo=pytz.utc)
    retrofit_completion_date = datetime(2013,8,1,tzinfo=pytz.utc)

    periods = generate_periods(start,end,jitter_intensity=0)
    gen = ConsumptionGenerator("electricity", "kWh", request.param[4], model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    consumption_kWh_per_day = [c.kWh / c.timedelta.days for c in consumptions]
    consumption_n_days = [p.timedelta.days for p in periods]
    fixture = ConsumptionHistory(consumptions), model.param_dict_to_list(params), \
            request.param[1], request.param[2], \
            request.param[3], request.param[4], \
            retrofit_start_date, retrofit_completion_date, \
            request.param[5], request.param[6], \
            request.param[7], request.param[8], \
            request.param[9], request.param[10], \
            consumption_kWh_per_day, consumption_kWh_per_day, \
            consumption_n_days, \
            request.param[11]
    return fixture

@pytest.mark.slow
def test_princeton_scorekeeping_method(prism_outputs_1,
                                       gsod_722880_2012_2014_weather_source,
                                       tmy3_722880_weather_source):
    ch, elec_params, elec_presence, \
            elec_annualized_usage, elec_error, temp_unit, \
            retrofit_start_date, retrofit_completion_date, \
            cdd_tmy, hdd_tmy, total_cdd, total_hdd, \
            rmse_electricity, r_squared_electricity, \
            average_daily_usages, estimated_average_daily_usages, n_days, \
            gas_param_defaults \
            = prism_outputs_1

    meter = PRISMMeter(temperature_unit_str=temp_unit)

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)

    assert_allclose(result['annualized_usage_electricity'], elec_annualized_usage, rtol=RTOL, atol=ATOL)
    assert_allclose(result['average_daily_usages_bpi2400_electricity'],average_daily_usages,rtol=RTOL,atol=ATOL)
    assert_allclose(result['average_daily_usages_bpi2400_natural_gas'],[],rtol=RTOL,atol=ATOL)
    assert_allclose(result['average_daily_usages_electricity'],average_daily_usages,rtol=RTOL,atol=ATOL)
    assert_allclose(result['cdd_tmy'], cdd_tmy, rtol=RTOL, atol=ATOL)
    assert result['consumption_history_no_estimated'] is not None
    assert result['cvrmse_electricity'] < 1e-2
    assert np.isnan(result['cvrmse_natural_gas'])
    assert result['electricity_presence'] == True
    assert_allclose(result['estimated_average_daily_usages_bpi2400_electricity'],estimated_average_daily_usages,rtol=RTOL,atol=ATOL)
    assert_allclose(result['estimated_average_daily_usages_bpi2400_natural_gas'],[],rtol=RTOL,atol=ATOL)
    assert_allclose(result['estimated_average_daily_usages_electricity'],estimated_average_daily_usages,rtol=RTOL,atol=ATOL)
    assert result['has_enough_cdd_electricity'] == True
    assert result['has_enough_cdd_natural_gas'] == True
    assert result['has_enough_data_electricity'] == True
    assert result['has_enough_data_natural_gas'] == False
    assert result['has_enough_hdd_cdd_electricity'] == True
    assert result['has_enough_hdd_cdd_natural_gas'] == False
    assert result['has_enough_hdd_electricity'] == True
    assert result['has_enough_hdd_natural_gas'] == False
    assert result['has_enough_periods_with_high_cdd_per_day_electricity'] == True
    assert result['has_enough_periods_with_high_cdd_per_day_natural_gas'] == True
    assert result['has_enough_periods_with_high_hdd_per_day_electricity'] == True
    assert result['has_enough_periods_with_high_hdd_per_day_natural_gas'] == False
    assert result['has_enough_periods_with_low_cdd_per_day_electricity'] == True
    assert result['has_enough_periods_with_low_cdd_per_day_natural_gas'] == True
    assert result['has_enough_periods_with_low_hdd_per_day_electricity'] == True
    assert result['has_enough_periods_with_low_hdd_per_day_natural_gas'] == False
    assert result['has_enough_total_cdd_electricity'] == True
    assert result['has_enough_total_cdd_natural_gas'] == True
    assert result['has_enough_total_hdd_electricity'] == True
    assert result['has_enough_total_hdd_natural_gas'] == False
    assert result['has_recent_reading_electricity'] == True
    assert result['has_recent_reading_natural_gas'] == False
    assert_allclose(result['hdd_tmy'], hdd_tmy, rtol=RTOL, atol=ATOL)
    assert result['meets_cvrmse_limit_electricity'] == True
    assert result['meets_cvrmse_limit_natural_gas'] == False
    assert result['meets_model_calibration_utility_bill_criteria_electricity'] == True
    assert result['meets_model_calibration_utility_bill_criteria_natural_gas'] == False
    assert_allclose(result['n_days_electricity'],n_days,rtol=RTOL,atol=ATOL)
    assert result['n_periods_high_cdd_per_day_electricity'] > 0
    assert result['n_periods_high_hdd_per_day_electricity'] > 0
    assert result['n_periods_low_cdd_per_day_electricity'] > 0
    assert result['n_periods_low_hdd_per_day_electricity'] > 0
    assert result['n_periods_high_cdd_per_day_natural_gas'] == 0
    assert result['n_periods_high_hdd_per_day_natural_gas'] == 0
    assert result['n_periods_low_cdd_per_day_natural_gas'] == 0
    assert result['n_periods_low_hdd_per_day_natural_gas'] == 0
    assert result['natural_gas_presence'] == False
    assert_allclose(result['rmse_electricity'], rmse_electricity, rtol=RTOL, atol=ATOL)
    assert_allclose(result['r_squared_electricity'], r_squared_electricity, rtol=RTOL, atol=ATOL)
    assert result['spans_183_days_and_has_enough_hdd_cdd_electricity'] == True
    assert result['spans_183_days_and_has_enough_hdd_cdd_natural_gas'] == False
    assert result['spans_184_days_electricity'] == True
    assert result['spans_184_days_natural_gas'] == False
    assert result['spans_330_days_electricity'] == True
    assert result['spans_330_days_natural_gas'] == False
    assert_allclose(result['temp_sensitivity_params_bpi2400_electricity'], elec_params, rtol=RTOL, atol=ATOL)
    assert_allclose(result['temp_sensitivity_params_bpi2400_natural_gas'], gas_param_defaults, rtol=RTOL, atol=ATOL)
    assert_allclose(result['temp_sensitivity_params_electricity'], elec_params, rtol=RTOL, atol=ATOL)
    assert result['time_span_electricity'] == 1080
    assert result['time_span_natural_gas'] == 0
    assert_allclose(result['total_cdd_electricity'], total_cdd, rtol=RTOL, atol=ATOL)
    assert result['total_cdd_natural_gas'] == 0
    assert_allclose(result['total_hdd_electricity'], total_hdd, rtol=RTOL, atol=ATOL)
    assert result['total_hdd_natural_gas'] == 0

    all_inputs = ["annualized_usage_electricity",
            "average_daily_usages_bpi2400_electricity",
            "average_daily_usages_bpi2400_natural_gas",
            "average_daily_usages_electricity",
            "cdd_tmy",
            "consumption_history_no_estimated",
            "cvrmse_electricity",
            "cvrmse_natural_gas",
            "electricity_presence",
            "estimated_average_daily_usages_bpi2400_electricity",
            "estimated_average_daily_usages_bpi2400_natural_gas",
            "estimated_average_daily_usages_electricity",
            "has_enough_cdd_electricity",
            "has_enough_cdd_natural_gas",
            "has_enough_data_electricity",
            "has_enough_data_natural_gas",
            "has_enough_hdd_cdd_electricity",
            "has_enough_hdd_cdd_natural_gas",
            "has_enough_hdd_electricity",
            "has_enough_hdd_natural_gas",
            "has_enough_periods_with_high_cdd_per_day_electricity",
            "has_enough_periods_with_high_cdd_per_day_natural_gas",
            "has_enough_periods_with_high_hdd_per_day_electricity",
            "has_enough_periods_with_high_hdd_per_day_natural_gas",
            "has_enough_periods_with_low_cdd_per_day_electricity",
            "has_enough_periods_with_low_cdd_per_day_natural_gas",
            "has_enough_periods_with_low_hdd_per_day_electricity",
            "has_enough_periods_with_low_hdd_per_day_natural_gas",
            "has_enough_total_cdd_electricity",
            "has_enough_total_cdd_natural_gas",
            "has_enough_total_hdd_electricity",
            "has_enough_total_hdd_natural_gas",
            "has_recent_reading_electricity",
            "has_recent_reading_natural_gas",
            "hdd_tmy",
            "meets_cvrmse_limit_electricity",
            "meets_cvrmse_limit_natural_gas",
            "meets_model_calibration_utility_bill_criteria_electricity",
            "meets_model_calibration_utility_bill_criteria_natural_gas",
            "n_days_electricity",
            "n_periods_high_cdd_per_day_electricity",
            "n_periods_high_hdd_per_day_electricity",
            "n_periods_low_cdd_per_day_electricity",
            "n_periods_low_hdd_per_day_electricity",
            "n_periods_high_cdd_per_day_natural_gas",
            "n_periods_high_hdd_per_day_natural_gas",
            "n_periods_low_cdd_per_day_natural_gas",
            "n_periods_low_hdd_per_day_natural_gas",
            "natural_gas_presence",
            "rmse_electricity",
            "r_squared_electricity",
            "spans_183_days_and_has_enough_hdd_cdd_electricity",
            "spans_183_days_and_has_enough_hdd_cdd_natural_gas",
            "spans_184_days_electricity",
            "spans_184_days_natural_gas",
            "spans_330_days_electricity",
            "spans_330_days_natural_gas",
            "temp_sensitivity_params_bpi2400_electricity",
            "temp_sensitivity_params_bpi2400_natural_gas",
            "temp_sensitivity_params_electricity",
            "time_span_electricity",
            "time_span_natural_gas",
            "total_cdd_electricity",
            "total_cdd_natural_gas",
            "total_hdd_electricity",
            "total_hdd_natural_gas"]
    assert set(all_inputs) == set(result.keys())

def test_prism_bad_inputs():

    with pytest.raises(ValueError):
        PRISMMeter(temperature_unit_str="unexpected")

    with pytest.raises(ValueError):
        PRISMMeter(heating_ref_temp_high=0,heating_ref_temp_x0=1,heating_ref_temp_low=2)

    with pytest.raises(ValueError):
        PRISMMeter(cooling_ref_temp_high=0,cooling_ref_temp_x0=1,cooling_ref_temp_low=2)

    with pytest.raises(ValueError):
        PRISMMeter(electricity_heating_slope_high=-1)

    with pytest.raises(ValueError):
        PRISMMeter(natural_gas_heating_slope_high=-1)

    with pytest.raises(ValueError):
        PRISMMeter(electricity_cooling_slope_high=-1)

