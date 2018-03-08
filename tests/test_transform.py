from datetime import datetime
from pkg_resources import resource_stream

import pandas as pd
import pytest

from eemeter import (
    billing_as_daily,
    get_baseline_data,
    get_reporting_data,
    merge_temperature_data,
    day_counts,
    NoBaselineDataError,
    NoReportingDataError,
)


def test_merge_temperature_data_hourly_temp_mean(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']['2016-03-01':'2016-07-01']
    temperature_data = il_electricity_cdd_hdd_hourly['temperature_data']['2016-03-01':'2016-07-01']
    df = merge_temperature_data(
        meter_data, temperature_data)
    assert df.shape == (2952, 2)
    assert list(df.columns) == ['meter_value', 'temperature_mean']
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 2914.0
    assert round(df.temperature_mean.mean()) == 62.0


def test_merge_temperature_data_hourly_hourly_degree_days(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']['2016-03-01':'2016-07-01']
    temperature_data = il_electricity_cdd_hdd_hourly['temperature_data']['2016-03-01':'2016-07-01']
    df = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method='hourly',
    )
    assert df.shape == (2952, 7)
    assert list(sorted(df.columns)) == [
        'cdd_65', 'cdd_66', 'hdd_60', 'hdd_61', 'meter_value', 'n_hours_dropped',
        'n_hours_kept',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 2914.0
    assert round(df.hdd_60.mean(), 2) == 5.25
    assert round(df.hdd_61.mean(), 2) == 5.72
    assert round(df.cdd_65.mean(), 2) == 4.74
    assert round(df.cdd_66.mean(), 2) == 4.33
    assert round(df.n_hours_kept.mean(), 2) == 1.0
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_hourly_daily_degree_days_fail(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']['2016-03-01':'2016-07-01']
    temperature_data = il_electricity_cdd_hdd_hourly['temperature_data']['2016-03-01':'2016-07-01']

    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data, temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method='daily',
        )


def test_merge_temperature_data_hourly_bad_degree_days(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']['2016-03-01':'2016-07-01']
    temperature_data = il_electricity_cdd_hdd_hourly['temperature_data']['2016-03-01':'2016-07-01']

    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data, temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method='UNKNOWN',
        )


def test_merge_temperature_data_hourly_data_quality(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']['2016-03-01':'2016-07-01']
    temperature_data = il_electricity_cdd_hdd_hourly['temperature_data']['2016-03-01':'2016-07-01']

    df = merge_temperature_data(
        meter_data, temperature_data, temperature_mean=False,
        data_quality=True,
    )
    assert df.shape == (2952, 3)
    assert list(df.columns) == [
        'meter_value', 'temperature_not_null', 'temperature_null',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 2914.0
    assert round(df.temperature_not_null.mean(), 2) == 1.0
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_merge_temperature_data_daily_temp_mean(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (810, 2)
    assert list(df.columns) == ['meter_value', 'temperature_mean']
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.temperature_mean.mean()) == 55.0


def test_merge_temperature_data_daily_daily_degree_days(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method='daily',
    )
    assert df.shape == (810, 7)
    assert list(sorted(df.columns)) == [
        'cdd_65', 'cdd_66', 'hdd_60', 'hdd_61', 'meter_value', 'n_days_dropped',
        'n_days_kept',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.hdd_60.mean(), 2) == 11.05
    assert round(df.hdd_61.mean(), 2) == 11.61
    assert round(df.cdd_65.mean(), 2) == 3.61
    assert round(df.cdd_66.mean(), 2) == 3.25
    assert round(df.n_days_kept.mean(), 2) == 1
    assert round(df.n_days_dropped.mean(), 2) == 0


def test_merge_temperature_data_daily_hourly_degree_days(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method='hourly',
    )
    assert df.shape == (810, 7)
    assert list(sorted(df.columns)) == [
        'cdd_65', 'cdd_66', 'hdd_60', 'hdd_61', 'meter_value', 'n_hours_dropped',
        'n_hours_kept',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.hdd_60.mean(), 2) == 11.48
    assert round(df.hdd_61.mean(), 2) == 12.06
    assert round(df.cdd_65.mean(), 2) == 4.04
    assert round(df.cdd_66.mean(), 2) == 3.69
    assert round(df.n_hours_kept.mean(), 2) == 23.97
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_daily_bad_degree_days(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data, temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method='UNKNOWN',
        )


def test_merge_temperature_data_daily_data_quality(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        temperature_mean=False,
        data_quality=True,
    )
    assert df.shape == (810, 3)
    assert list(df.columns) == [
        'meter_value', 'temperature_not_null', 'temperature_null',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.temperature_not_null.mean(), 2) == 23.97
    assert round(df.temperature_null.mean(), 2) == 0.00


def test_merge_temperature_data_billing_monthly_temp_mean(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_monthly['temperature_data']
    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (27, 2)
    assert list(df.columns) == ['meter_value', 'temperature_mean']
    assert round(meter_data.value.sum()) == round(df.meter_value.sum()) == 21290.0
    assert round(df.temperature_mean.mean()) == 54.0


def test_merge_temperature_data_billing_monthly_daily_degree_days(
        il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_monthly['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method='daily',
    )
    assert df.shape == (27, 7)
    assert list(sorted(df.columns)) == [
        'cdd_65', 'cdd_66', 'hdd_60', 'hdd_61', 'meter_value', 'n_days_dropped',
        'n_days_kept',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21290.0
    assert round(df.hdd_60.mean(), 2) == 11.42
    assert round(df.hdd_61.mean(), 2) == 12.0
    assert round(df.cdd_65.mean(), 2) == 3.54
    assert round(df.cdd_66.mean(), 2) == 3.19
    assert round(df.n_days_kept.mean(), 2) == 29.96
    assert round(df.n_days_dropped.mean(), 2) == 0.04


def test_merge_temperature_data_billing_monthly_hourly_degree_days(
        il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_monthly['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method='hourly',
    )
    assert df.shape == (27, 7)
    assert list(sorted(df.columns)) == [
        'cdd_65', 'cdd_66', 'hdd_60', 'hdd_61', 'meter_value', 'n_hours_dropped',
        'n_hours_kept',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21290.0
    assert round(df.hdd_60.mean(), 2) == 11.8
    assert round(df.hdd_61.mean(), 2) == 12.38
    assert round(df.cdd_65.mean(), 2) == 3.96
    assert round(df.cdd_66.mean(), 2) == 3.62
    assert round(df.n_hours_kept.mean(), 2) == 719.15
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_billing_monthly_bad_degree_day_method(
        il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_monthly['temperature_data']
    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data, temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method='UNKNOWN',
        )


def test_merge_temperature_data_billing_monthly_data_quality(
        il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_monthly['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        temperature_mean=False,
        data_quality=True,
    )
    assert df.shape == (27, 3)
    assert list(df.columns) == [
        'meter_value', 'temperature_not_null', 'temperature_null',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21290.0
    assert round(df.temperature_not_null.mean(), 2) == 719.15
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_merge_temperature_data_billing_bimonthly_temp_mean(
        il_electricity_cdd_hdd_billing_bimonthly):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly['temperature_data']
    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (14, 2)
    assert list(df.columns) == ['meter_value', 'temperature_mean']
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21564.0
    assert round(df.temperature_mean.mean()) == 53.0


def test_merge_temperature_data_billing_bimonthly_daily_degree_days(
        il_electricity_cdd_hdd_billing_bimonthly):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method='daily',
    )
    assert df.shape == (14, 7)
    assert list(sorted(df.columns)) == [
        'cdd_65', 'cdd_66', 'hdd_60', 'hdd_61', 'meter_value', 'n_days_dropped',
        'n_days_kept',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21564.0
    assert round(df.hdd_60.mean(), 2) == 12.72
    assert round(df.hdd_61.mean(), 2) == 13.32
    assert round(df.cdd_65.mean(), 2) == 3.39
    assert round(df.cdd_66.mean(), 2) == 3.05
    assert round(df.n_days_kept.mean(), 2) == 57.79
    assert round(df.n_days_dropped.mean(), 2) == 0.07


def test_merge_temperature_data_billing_bimonthly_hourly_degree_days(
        il_electricity_cdd_hdd_billing_bimonthly):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method='hourly',
    )
    assert df.shape == (14, 7)
    assert list(sorted(df.columns)) == [
        'cdd_65', 'cdd_66', 'hdd_60', 'hdd_61', 'meter_value', 'n_hours_dropped',
        'n_hours_kept',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21564.0
    assert round(df.hdd_60.mean(), 2) == 13.08
    assert round(df.hdd_61.mean(), 2) == 13.69
    assert round(df.cdd_65.mean(), 2) == 3.78
    assert round(df.cdd_66.mean(), 2) == 3.46
    assert round(df.n_hours_kept.mean(), 2) == 1386.93
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_billing_bimonthly_bad_degree_days(
        il_electricity_cdd_hdd_billing_bimonthly):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly['temperature_data']
    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data, temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method='UNKNOWN',
        )


def test_merge_temperature_data_billing_bimonthly_data_quality(
        il_electricity_cdd_hdd_billing_bimonthly):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly['meter_data']
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly['temperature_data']
    df = merge_temperature_data(
        meter_data, temperature_data,
        temperature_mean=False,
        data_quality=True,
    )
    assert df.shape == (14, 3)
    assert list(df.columns) == [
        'meter_value', 'temperature_not_null', 'temperature_null',
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21564.0
    assert round(df.temperature_not_null.mean(), 2) == 1386.93
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_billing_as_daily(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly['meter_data']
    assert meter_data.shape == (27, 1)
    as_daily = billing_as_daily(meter_data)
    assert as_daily.shape == (791, 1)
    assert round(meter_data.value.sum(), 1) == round(as_daily.value.sum(), 1) == 21290.2


def test_day_counts(il_electricity_cdd_hdd_billing_monthly):
    data = il_electricity_cdd_hdd_billing_monthly['meter_data'].value
    counts = day_counts(data)
    assert counts.shape == (27,)
    assert counts.iloc[0] == 29.0
    assert pd.isnull(counts.iloc[-1])
    assert counts.sum() == 790.0


def test_get_baseline_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    baseline_data = get_baseline_data(meter_data)
    assert meter_data.shape == baseline_data.shape == (19417, 1)


def test_get_baseline_data_with_end(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    blackout_start_date = il_electricity_cdd_hdd_hourly['blackout_start_date']
    baseline_data = get_baseline_data(meter_data, end=blackout_start_date)
    assert meter_data.shape != baseline_data.shape == (8761, 1)


def test_get_baseline_data_with_end_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    blackout_start_date = il_electricity_cdd_hdd_hourly['blackout_start_date']
    baseline_data = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=None)
    assert meter_data.shape != baseline_data.shape == (9595, 1)


def test_get_baseline_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    blackout_start_date = il_electricity_cdd_hdd_hourly['blackout_start_date']
    with pytest.raises(NoBaselineDataError):
        get_baseline_data(meter_data, end=pd.Timestamp('2000').tz_localize('UTC'))


def test_get_reporting_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    reporting_data = get_reporting_data(meter_data)
    assert meter_data.shape == reporting_data.shape == (19417, 1)


def test_get_reporting_data_with_start(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    blackout_end_date = il_electricity_cdd_hdd_hourly['blackout_end_date']
    reporting_data = get_reporting_data(
        meter_data, start=blackout_end_date)
    assert meter_data.shape != reporting_data.shape == (8761, 1)


def test_get_reporting_data_with_start_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    blackout_end_date = il_electricity_cdd_hdd_hourly['blackout_end_date']
    reporting_data = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=None)
    assert meter_data.shape != reporting_data.shape == (9607, 1)


def test_get_reporting_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly['meter_data']
    blackout_end_date = il_electricity_cdd_hdd_hourly['blackout_end_date']
    with pytest.raises(NoReportingDataError):
        get_reporting_data(meter_data, start=pd.Timestamp('2030').tz_localize('UTC'))
