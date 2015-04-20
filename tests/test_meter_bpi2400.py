from eemeter.meter import BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria
from fixtures.weather import tmy3_722880_weather_source
from fixtures.weather import gsod_722880_2012_2014_weather_source

from eemeter.models import TemperatureSensitivityModel
from eemeter.generator import generate_periods,ConsumptionGenerator
from eemeter.consumption import ConsumptionHistory

from datetime import datetime

import pytest

from numpy.testing import assert_allclose

RTOL = 1e-2
ATOL = 1e-2

@pytest.fixture(params=[([10,2,58,1,72],[10,1,65],1248.4575,1578.5882,0,0,
                         [15,15,7,7,11,11,16,16],1080,4657.6997,3249.0001,"degF"),
                        ([10,2,14.4,1,22.22],[10,1,18.33],693.5875,876.9934,0.0573,0,
                         [15,15,7,7,11,11,16,16],1080,2587.610,1805.0001,"degC")])
def bpi_2400_1(request):
    elec_params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    gas_params = {
        "base_consumption": request.param[1][0],
        "heating_slope": request.param[1][1],
        "heating_reference_temperature": request.param[1][2],
    }

    normal_cdd = request.param[2]
    normal_hdd = request.param[3]
    cvrmse_electricity = request.param[4]
    cvrmse_natural_gas = request.param[5]
    n_periods = request.param[6]
    time_span = request.param[7]
    total_cdd = request.param[8]
    total_hdd = request.param[9]
    temp_unit = request.param[10]

    start = datetime(2012,1,1)
    end = datetime(2014,12,31)
    periods = generate_periods(start,end,jitter_intensity=0)
    elec_model = TemperatureSensitivityModel(cooling=True,heating=True)
    gas_model = TemperatureSensitivityModel(cooling=False,heating=True)
    gen_elec = ConsumptionGenerator("electricity", "kWh", temp_unit, elec_model, elec_params)
    gen_gas = ConsumptionGenerator("natural_gas", "therms", temp_unit, gas_model, gas_params)
    elec_consumptions = gen_elec.generate(gsod_722880_2012_2014_weather_source(), periods)
    gas_consumptions = gen_gas.generate(gsod_722880_2012_2014_weather_source(), periods)
    ch = ConsumptionHistory(elec_consumptions + gas_consumptions)

    elec_param_list = elec_model.param_dict_to_list(elec_params)
    gas_param_list = gas_model.param_dict_to_list(gas_params)

    return ch, elec_param_list, gas_param_list, normal_cdd, normal_hdd, \
            cvrmse_electricity, cvrmse_natural_gas, \
            n_periods, time_span, total_cdd, total_hdd, temp_unit

def test_bpi2400(bpi_2400_1,
                 gsod_722880_2012_2014_weather_source,
                 tmy3_722880_weather_source):

    ch, elec_params, gas_params, normal_cdd, normal_hdd, \
            cvrmse_electricity, cvrmse_natural_gas, n_periods, time_span,\
            total_cdd, total_hdd, temp_unit = bpi_2400_1

    meter = BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria(temperature_unit_str=temp_unit)

    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)

    assert 'consumption_history' not in result
    assert 'weather_normal_source' not in result
    assert 'weather_source' not in result

    for c in result["consumption_history_no_estimated"].iteritems():
        assert not c.estimated

    assert_allclose(result['cdd_tmy'],normal_cdd,rtol=RTOL,atol=ATOL)
    assert_allclose(result['hdd_tmy'],normal_hdd,rtol=RTOL,atol=ATOL)

    assert_allclose(result['cvrmse_electricity'],cvrmse_electricity,rtol=RTOL,atol=ATOL)
    assert_allclose(result['cvrmse_natural_gas'],cvrmse_natural_gas,rtol=RTOL,atol=ATOL)

    assert result['electricity_presence']
    assert result['natural_gas_presence']

    assert result['has_enough_cdd_electricity']
    assert result['has_enough_cdd_natural_gas']
    assert result['has_enough_data_electricity']
    assert result['has_enough_data_natural_gas']
    assert result['has_enough_hdd_cdd_electricity']
    assert result['has_enough_hdd_cdd_natural_gas']
    assert result['has_enough_hdd_electricity']
    assert result['has_enough_hdd_natural_gas']
    assert result['has_enough_periods_with_high_cdd_per_day_electricity']
    assert result['has_enough_periods_with_high_cdd_per_day_natural_gas']
    assert result['has_enough_periods_with_high_hdd_per_day_electricity']
    assert result['has_enough_periods_with_high_hdd_per_day_natural_gas']
    assert result['has_enough_periods_with_low_cdd_per_day_electricity']
    assert result['has_enough_periods_with_low_cdd_per_day_natural_gas']
    assert result['has_enough_periods_with_low_hdd_per_day_electricity']
    assert result['has_enough_periods_with_low_hdd_per_day_natural_gas']
    assert result['has_enough_total_cdd_electricity']
    assert result['has_enough_total_cdd_natural_gas']
    assert result['has_enough_total_hdd_electricity']
    assert result['has_enough_total_hdd_natural_gas']
    assert result['has_recent_reading_electricity']
    assert result['has_recent_reading_natural_gas']
    assert result['meets_cvrmse_limit_electricity']
    assert result['meets_cvrmse_limit_natural_gas']
    assert result['meets_model_calibration_utility_bill_criteria_electricity']
    assert result['meets_model_calibration_utility_bill_criteria_natural_gas']

    assert_allclose(result['n_periods_high_cdd_per_day_electricity'],n_periods[0],rtol=RTOL,atol=ATOL)
    assert_allclose(result['n_periods_high_cdd_per_day_natural_gas'],n_periods[1],rtol=RTOL,atol=ATOL)
    assert_allclose(result['n_periods_high_hdd_per_day_electricity'],n_periods[2],rtol=RTOL,atol=ATOL)
    assert_allclose(result['n_periods_high_hdd_per_day_natural_gas'],n_periods[3],rtol=RTOL,atol=ATOL)
    assert_allclose(result['n_periods_low_cdd_per_day_electricity'],n_periods[4],rtol=RTOL,atol=ATOL)
    assert_allclose(result['n_periods_low_cdd_per_day_natural_gas'],n_periods[5],rtol=RTOL,atol=ATOL)
    assert_allclose(result['n_periods_low_hdd_per_day_electricity'],n_periods[6],rtol=RTOL,atol=ATOL)
    assert_allclose(result['n_periods_low_hdd_per_day_natural_gas'],n_periods[7],rtol=RTOL,atol=ATOL)

    assert result['spans_183_days_and_has_enough_hdd_cdd_electricity']
    assert result['spans_183_days_and_has_enough_hdd_cdd_natural_gas']
    assert result['spans_184_days_electricity']
    assert result['spans_184_days_natural_gas']
    assert result['spans_330_days_electricity']
    assert result['spans_330_days_natural_gas']

    assert_allclose(result['time_span_electricity'],time_span,rtol=RTOL,atol=ATOL)
    assert_allclose(result['time_span_natural_gas'],time_span,rtol=RTOL,atol=ATOL)

    assert_allclose(result['total_cdd_electricity'],total_cdd,rtol=RTOL,atol=ATOL)
    assert_allclose(result['total_cdd_natural_gas'],total_cdd,rtol=RTOL,atol=ATOL)
    assert_allclose(result['total_hdd_electricity'],total_hdd,rtol=RTOL,atol=ATOL)
    assert_allclose(result['total_hdd_natural_gas'],total_hdd,rtol=RTOL,atol=ATOL)
