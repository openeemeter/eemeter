from eemeter.meter import BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria
from fixtures.weather import tmy3_722880_weather_source
from fixtures.weather import gsod_722880_2012_2014_weather_source

from eemeter.meter import DataCollection

from eemeter.models import AverageDailyTemperatureSensitivityModel
from eemeter.generator import generate_monthly_billing_datetimes
from eemeter.generator import MonthlyBillingConsumptionGenerator
from eemeter.consumption import ConsumptionData
from eemeter.evaluation import Period

from datetime import datetime

import pytest

from numpy.testing import assert_allclose
from scipy.stats import randint
import pytz

RTOL = 1e-2
ATOL = 1e-2

@pytest.fixture(params=[([10,58,2,72,1],[10,65,1],1248.4575,1578.5882,0,0,
                         [15,15,7,7,11,11,16,16],1080,4657.6997,3249.0001,"degF"),
                        ([10,14.4,2,22.22,1],[10,18.33,1],693.5875,876.9934,0.0573,0,
                         [15,15,7,7,11,11,16,16],1080,2587.610,1805.0001,"degC")])
def bpi_2400_1(request,gsod_722880_2012_2014_weather_source):
    elec_param_list, gas_param_list, normal_cdd, normal_hdd, \
            cvrmse_electricity, cvrmse_natural_gas, n_periods, time_span, \
            total_cdd, total_hdd, temp_unit = request.param

    elec_params = {
        "base_daily_consumption": elec_param_list[0],
        "heating_balance_temperature": elec_param_list[1],
        "heating_slope": elec_param_list[2],
        "cooling_balance_temperature": elec_param_list[3],
        "cooling_slope": elec_param_list[4],
    }

    gas_params = {
        "base_daily_consumption": gas_param_list[0],
        "heating_balance_temperature": gas_param_list[1],
        "heating_slope": gas_param_list[2],
    }

    period = Period(datetime(2012, 1, 1, tzinfo=pytz.utc),
            datetime(2014, 12, 31, tzinfo=pytz.utc))
    datetimes = generate_monthly_billing_datetimes(period, randint(30,31))
    elec_model = AverageDailyTemperatureSensitivityModel(cooling=True, heating=True)
    gas_model = AverageDailyTemperatureSensitivityModel(cooling=False, heating=True)
    gen_elec = MonthlyBillingConsumptionGenerator("electricity", "kWh", temp_unit, elec_model, elec_params)
    gen_gas = MonthlyBillingConsumptionGenerator("natural_gas", "therm", temp_unit, gas_model, gas_params)
    elec_consumptions = gen_elec.generate(gsod_722880_2012_2014_weather_source, datetimes)
    gas_consumptions = gen_gas.generate(gsod_722880_2012_2014_weather_source, datetimes)

    average_daily_usages_elec = elec_consumptions.average_daily_consumptions()[0]
    average_daily_usages_gas = gas_consumptions.average_daily_consumptions()[0]

    return elec_consumptions, gas_consumptions, elec_param_list,\
            gas_param_list, normal_cdd, normal_hdd, \
            cvrmse_electricity, cvrmse_natural_gas, \
            n_periods, time_span, total_cdd, total_hdd, temp_unit, \
            average_daily_usages_elec, average_daily_usages_gas

def test_bpi2400(bpi_2400_1,
                 gsod_722880_2012_2014_weather_source,
                 tmy3_722880_weather_source):

    elec_consumptions, gas_consumptions, elec_params, gas_params, normal_cdd, \
            normal_hdd, cvrmse_electricity, cvrmse_natural_gas, n_periods, \
            time_span, total_cdd, total_hdd, temp_unit, \
            average_daily_usages_electricity, \
            average_daily_usages_natural_gas \
            = bpi_2400_1

    meter = BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria(temperature_unit_str=temp_unit)

    data_elec = DataCollection(
            consumption_data=elec_consumptions,
            weather_source=gsod_722880_2012_2014_weather_source,
            weather_normal_source=tmy3_722880_weather_source)
    result_elec = meter.evaluate(data_elec)

    data_gas = DataCollection(
            consumption_data=gas_consumptions,
            weather_source=gsod_722880_2012_2014_weather_source,
            weather_normal_source=tmy3_722880_weather_source)
    result_gas = meter.evaluate(data_gas)

    elec = [result_elec, elec_params, average_daily_usages_electricity, cvrmse_electricity]
    gas = [result_gas, gas_params, average_daily_usages_natural_gas, cvrmse_natural_gas]

    for result, params, average_daily_usages, cvrmse in [elec, gas]:

        assert_allclose(result_elec.get_data('cdd_tmy').value, normal_cdd,
                rtol=RTOL, atol=ATOL)
        assert_allclose(result_elec.get_data('hdd_tmy').value, normal_hdd,
                rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data("temp_sensitivity_params_bpi2400").value.to_list(),
                params, rtol=RTOL,atol=ATOL)
        assert_allclose(result.get_data('average_daily_usages_bpi2400').value,
                average_daily_usages, rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data('cvrmse').value,
                cvrmse, rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data("estimated_average_daily_usages_bpi2400").value,
                average_daily_usages, rtol=RTOL, atol=ATOL)


    for result in [result_elec, result_gas]:
        assert result.get_data('has_enough_cdd').value
        assert result.get_data('has_enough_data').value
        assert result.get_data('has_enough_hdd_cdd').value
        assert result.get_data('has_enough_hdd').value
        assert result.get_data('has_enough_periods_with_high_cdd_per_day').value
        assert result.get_data('has_enough_periods_with_high_hdd_per_day').value
        assert result.get_data('has_enough_periods_with_low_cdd_per_day').value
        assert result.get_data('has_enough_periods_with_low_hdd_per_day').value
        assert result.get_data('has_enough_total_cdd').value
        assert result.get_data('has_enough_total_hdd').value
        assert result.get_data('has_recent_reading').value

        assert result.get_data('meets_cvrmse_limit').value
        assert result.get_data('meets_model_calibration_utility_bill_criteria').value

        assert_allclose(result.get_data('n_periods_high_cdd_per_day').value,
                n_periods[0], rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data('n_periods_high_hdd_per_day').value,
                n_periods[2], rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data('n_periods_low_cdd_per_day').value,
                n_periods[4], rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data('n_periods_low_hdd_per_day').value,
                n_periods[6], rtol=RTOL, atol=ATOL)

        assert result.get_data('spans_183_days_and_has_enough_hdd_cdd').value
        assert result.get_data('spans_184_days').value
        assert result.get_data('spans_330_days').value

        assert_allclose(result.get_data('time_span').value, time_span,
                rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data('total_cdd').value, total_cdd,
                rtol=RTOL, atol=ATOL)
        assert_allclose(result.get_data('total_hdd').value, total_hdd,
                rtol=RTOL, atol=ATOL)
