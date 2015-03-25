from eemeter.meter import BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria
from fixtures.consumption import bpi_2400_1
from fixtures.weather import tmy3_722880_weather_source
from fixtures.weather import gsod_722880_2012_2014_weather_source

def test_bpi2400(bpi_2400_1,
                 gsod_722880_2012_2014_weather_source,
                 tmy3_722880_weather_source):

    meter = BPI_2400_S_2012_ModelCalibrationUtilityBillCriteria()
    ch, elec_params, gas_params = bpi_2400_1
    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)

    assert 'consumption_history' not in result
    assert 'weather_normal_source' not in result
    assert 'weather_source' not in result

    for c in result["consumption_history_no_estimated"].iteritems():
        assert not c.estimated

    assert 'cdd_65_tmy' in result
    assert 'hdd_65_tmy' in result

    assert 'cvrmse_electricity' in result
    assert 'cvrmse_natural_gas' in result

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

    assert 'n_periods_high_cdd_per_day_electricity' in result
    assert 'n_periods_high_cdd_per_day_natural_gas' in result
    assert 'n_periods_high_hdd_per_day_electricity' in result
    assert 'n_periods_high_hdd_per_day_natural_gas' in result
    assert 'n_periods_low_cdd_per_day_electricity' in result
    assert 'n_periods_low_cdd_per_day_natural_gas' in result
    assert 'n_periods_low_hdd_per_day_electricity' in result
    assert 'n_periods_low_hdd_per_day_natural_gas' in result

    assert result['spans_183_days_and_has_enough_hdd_cdd_electricity']
    assert result['spans_183_days_and_has_enough_hdd_cdd_natural_gas']
    assert result['spans_184_days_electricity']
    assert result['spans_184_days_natural_gas']
    assert result['spans_330_days_electricity']
    assert result['spans_330_days_natural_gas']

    assert 'time_span_electricity' in result
    assert 'time_span_natural_gas' in result
    assert 'total_cdd_electricity' in result
    assert 'total_cdd_natural_gas' in result
    assert 'total_hdd_electricity' in result
    assert 'total_hdd_natural_gas' in result
