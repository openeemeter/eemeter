from datetime import datetime, timedelta
from pkg_resources import resource_stream

import pandas as pd
import pytest

from eemeter import (
    as_freq,
    compute_temperature_features,
    day_counts,
    get_baseline_data,
    get_reporting_data,
    merge_temperature_data,
    remove_duplicates,
    NoBaselineDataError,
    NoReportingDataError,
)


def test_merge_temperature_data_no_freq_index(il_electricity_cdd_hdd_billing_monthly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_data.index.freq = None
    with pytest.raises(ValueError):
        merge_temperature_data(meter_data, temperature_data)


def test_compute_temperature_features_no_freq_index(
    il_electricity_cdd_hdd_billing_monthly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_data.index.freq = None
    with pytest.raises(ValueError):
        compute_temperature_features(temperature_data, meter_data.index)


def test_merge_temperature_data_no_meter_data_tz(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    meter_data = meter_data.tz_localize(None)
    with pytest.raises(ValueError):
        merge_temperature_data(meter_data, temperature_data)


def test_compute_temperature_features_no_meter_data_tz(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    meter_data.index = meter_data.index.tz_localize(None)
    with pytest.raises(ValueError):
        compute_temperature_features(temperature_data, meter_data.index)


def test_merge_temperature_data_no_temp_data_tz(il_electricity_cdd_hdd_billing_monthly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_data = temperature_data.tz_localize(None)
    with pytest.raises(ValueError):
        merge_temperature_data(meter_data, temperature_data)


def test_compute_temperature_features_no_temp_data_tz(
    il_electricity_cdd_hdd_billing_monthly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    temperature_data = temperature_data.tz_localize(None)
    with pytest.raises(ValueError):
        compute_temperature_features(temperature_data, meter_data.index)


def test_merge_temperature_data_hourly_temp_mean(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]
    df = merge_temperature_data(meter_data, temperature_data)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert df.shape == (2952, 4)

    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 2914.0
    assert round(df.temperature_mean.mean()) == 62.0


def test_compute_temperature_features_hourly_temp_mean(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]
    df = compute_temperature_features(temperature_data, meter_data.index)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert df.shape == (2952, 3)

    assert round(df.temperature_mean.mean()) == 62.0


def test_merge_temperature_data_hourly_hourly_degree_days(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert df.shape == (2952, 7)
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 2914.0
    assert round(df.hdd_60.mean(), 2) == 5.25
    assert round(df.hdd_61.mean(), 2) == 5.72
    assert round(df.cdd_65.mean(), 2) == 4.74
    assert round(df.cdd_66.mean(), 2) == 4.33
    assert round(df.n_hours_kept.mean(), 2) == 1.0
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_compute_temperature_features_hourly_hourly_degree_days(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert df.shape == (2952, 6)
    assert round(df.hdd_60.mean(), 2) == 5.25
    assert round(df.hdd_61.mean(), 2) == 5.72
    assert round(df.cdd_65.mean(), 2) == 4.74
    assert round(df.cdd_66.mean(), 2) == 4.33
    assert round(df.n_hours_kept.mean(), 2) == 1.0
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_hourly_hourly_degree_days_use_mean_false(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
        use_mean_daily_values=False,
    )
    assert df.shape == (2952, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 2914.0
    assert round(df.hdd_60.mean(), 2) == 0.22
    assert round(df.hdd_61.mean(), 2) == 0.24
    assert round(df.cdd_65.mean(), 2) == 0.2
    assert round(df.cdd_66.mean(), 2) == 0.18
    assert round(df.n_hours_kept.mean(), 2) == 1.0
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_compute_temperature_features_hourly_hourly_degree_days_use_mean_false(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
        use_mean_daily_values=False,
    )
    assert df.shape == (2952, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 0.22
    assert round(df.hdd_61.mean(), 2) == 0.24
    assert round(df.cdd_65.mean(), 2) == 0.2
    assert round(df.cdd_66.mean(), 2) == 0.18
    assert round(df.n_hours_kept.mean(), 2) == 1.0
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_hourly_daily_degree_days_fail(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data,
            temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="daily",
        )


def test_compute_temperature_features_hourly_daily_degree_days_fail(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    with pytest.raises(ValueError):
        compute_temperature_features(
            temperature_data,
            meter_data.index,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="daily",
        )


def test_merge_temperature_data_hourly_daily_missing_explicit_freq(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    meter_data.index.freq = None
    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data,
            temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="daily",
        )


def test_compute_temperature_features_hourly_daily_missing_explicit_freq(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    meter_data.index.freq = None
    with pytest.raises(ValueError):
        compute_temperature_features(
            temperature_data,
            meter_data.index,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="daily",
        )


def test_merge_temperature_data_hourly_bad_degree_days(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data,
            temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_compute_temperature_features_hourly_bad_degree_days(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    with pytest.raises(ValueError):
        compute_temperature_features(
            temperature_data,
            meter_data.index,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_merge_temperature_data_hourly_data_quality(il_electricity_cdd_hdd_hourly):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    df = merge_temperature_data(
        meter_data, temperature_data, temperature_mean=False, data_quality=True
    )
    assert df.shape == (2952, 5)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 2914.0
    assert round(df.temperature_not_null.mean(), 2) == 1.0
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_compute_temperature_features_hourly_data_quality(
    il_electricity_cdd_hdd_hourly
):
    # pick a slice with both hdd and cdd
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]["2016-03-01":"2016-07-01"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"][
        "2016-03-01":"2016-07-01"
    ]

    df = compute_temperature_features(
        temperature_data, meter_data.index, temperature_mean=False, data_quality=True
    )
    assert df.shape == (2952, 4)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.temperature_not_null.mean(), 2) == 1.0
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_merge_temperature_data_daily_temp_mean(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (810, 4)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]

    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.temperature_mean.mean()) == 55.0


def test_compute_temperature_features_daily_temp_mean(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = compute_temperature_features(temperature_data, meter_data.index)
    assert df.shape == (810, 3)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]

    assert round(df.temperature_mean.mean()) == 55.0


def test_merge_temperature_data_daily_daily_degree_days(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
    )
    assert df.shape == (810, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.hdd_60.mean(), 2) == 11.05
    assert round(df.hdd_61.mean(), 2) == 11.61
    assert round(df.cdd_65.mean(), 2) == 3.61
    assert round(df.cdd_66.mean(), 2) == 3.25
    assert round(df.n_days_kept.mean(), 2) == 1
    assert round(df.n_days_dropped.mean(), 2) == 0


def test_compute_temperature_features_daily_daily_degree_days(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
    )
    assert df.shape == (810, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 11.05
    assert round(df.hdd_61.mean(), 2) == 11.61
    assert round(df.cdd_65.mean(), 2) == 3.61
    assert round(df.cdd_66.mean(), 2) == 3.25
    assert round(df.n_days_kept.mean(), 2) == 1
    assert round(df.n_days_dropped.mean(), 2) == 0


def test_merge_temperature_data_daily_daily_degree_days_use_mean_false(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (810, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.hdd_60.mean(), 2) == 11.05
    assert round(df.hdd_61.mean(), 2) == 11.61
    assert round(df.cdd_65.mean(), 2) == 3.61
    assert round(df.cdd_66.mean(), 2) == 3.25
    assert round(df.n_days_kept.mean(), 2) == 1
    assert round(df.n_days_dropped.mean(), 2) == 0


def test_compute_temperature_features_daily_daily_degree_days_use_mean_false(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (810, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 11.05
    assert round(df.hdd_61.mean(), 2) == 11.61
    assert round(df.cdd_65.mean(), 2) == 3.61
    assert round(df.cdd_66.mean(), 2) == 3.25
    assert round(df.n_days_kept.mean(), 2) == 1
    assert round(df.n_days_dropped.mean(), 2) == 0


def test_merge_temperature_data_daily_hourly_degree_days(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert df.shape == (810, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.hdd_60.mean(), 2) == 11.48
    assert round(df.hdd_61.mean(), 2) == 12.06
    assert round(df.cdd_65.mean(), 2) == 4.04
    assert round(df.cdd_66.mean(), 2) == 3.69
    assert round(df.n_hours_kept.mean(), 2) == 23.97
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_compute_temperature_features_daily_hourly_degree_days(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert df.shape == (810, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 11.48
    assert round(df.hdd_61.mean(), 2) == 12.06
    assert round(df.cdd_65.mean(), 2) == 4.04
    assert round(df.cdd_66.mean(), 2) == 3.69
    assert round(df.n_hours_kept.mean(), 2) == 23.97
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_daily_hourly_degree_days_use_mean_false(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
        use_mean_daily_values=False,
    )
    assert df.shape == (810, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.hdd_60.mean(), 2) == 11.43
    assert round(df.hdd_61.mean(), 2) == 12.01
    assert round(df.cdd_65.mean(), 2) == 4.04
    assert round(df.cdd_66.mean(), 2) == 3.69
    assert round(df.n_hours_kept.mean(), 2) == 23.97
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_compute_temperature_features_daily_hourly_degree_days_use_mean_false(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
        use_mean_daily_values=False,
    )
    assert df.shape == (810, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 11.43
    assert round(df.hdd_61.mean(), 2) == 12.01
    assert round(df.cdd_65.mean(), 2) == 4.04
    assert round(df.cdd_66.mean(), 2) == 3.69
    assert round(df.n_hours_kept.mean(), 2) == 23.97
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_daily_bad_degree_days(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data,
            temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_compute_temperature_features_daily_bad_degree_days(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    with pytest.raises(ValueError):
        compute_temperature_features(
            temperature_data,
            meter_data.index,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_merge_temperature_data_daily_data_quality(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = merge_temperature_data(
        meter_data, temperature_data, temperature_mean=False, data_quality=True
    )
    assert df.shape == (810, 5)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21926.0
    assert round(df.temperature_not_null.mean(), 2) == 23.97
    assert round(df.temperature_null.mean(), 2) == 0.00


def test_compute_temperature_features_daily_data_quality(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    df = compute_temperature_features(
        temperature_data, meter_data.index, temperature_mean=False, data_quality=True
    )
    assert df.shape == (810, 4)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.temperature_not_null.mean(), 2) == 23.97
    assert round(df.temperature_null.mean(), 2) == 0.00


def test_merge_temperature_data_billing_monthly_temp_mean(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (27, 4)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.meter_value.sum()) == 703.0 != round(meter_data.value.sum())
    assert round(df.temperature_mean.mean()) == 55.0


def test_compute_temperature_features_billing_monthly_temp_mean(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = compute_temperature_features(temperature_data, meter_data.index)
    assert df.shape == (27, 3)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.temperature_mean.mean()) == 54.0


def test_merge_temperature_data_billing_monthly_daily_degree_days(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
    )
    assert df.shape == (27, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.meter_value.sum()) == 703.0 != round(meter_data.value.sum())
    assert round(df.hdd_60.mean(), 2) == 10.83
    assert round(df.hdd_61.mean(), 2) == 11.39
    assert round(df.cdd_65.mean(), 2) == 3.68
    assert round(df.cdd_66.mean(), 2) == 3.31
    assert round(df.n_days_kept.mean(), 2) == 30.38
    assert round(df.n_days_dropped.mean(), 2) == 0.00


def test_compute_temperature_features_billing_monthly_daily_degree_days(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
    )
    assert df.shape == (27, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 11.42
    assert round(df.hdd_61.mean(), 2) == 12.0
    assert round(df.cdd_65.mean(), 2) == 3.54
    assert round(df.cdd_66.mean(), 2) == 3.19
    assert round(df.n_days_kept.mean(), 2) == 29.96
    assert round(df.n_days_dropped.mean(), 2) == 0.04


def test_merge_temperature_data_billing_monthly_daily_degree_days_use_mean_false(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (27, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21290.0
    assert round(df.hdd_60.mean(), 2) == 324.38
    assert round(df.hdd_61.mean(), 2) == 341.38
    assert round(df.cdd_65.mean(), 2) == 112.59
    assert round(df.cdd_66.mean(), 2) == 101.33
    assert round(df.n_days_kept.mean(), 2) == 30.38
    assert round(df.n_days_dropped.mean(), 2) == 0.00


def test_compute_temperature_features_billing_monthly_daily_degree_days_use_mean_false(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (27, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 332.23
    assert round(df.hdd_61.mean(), 2) == 349.34
    assert round(df.cdd_65.mean(), 2) == 108.42
    assert round(df.cdd_66.mean(), 2) == 97.58
    assert round(df.n_days_kept.mean(), 2) == 29.96
    assert round(df.n_days_dropped.mean(), 2) == 0.04


def test_merge_temperature_data_billing_monthly_hourly_degree_days(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert df.shape == (27, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.meter_value.sum()) == 703.0 != round(meter_data.value.sum())
    assert round(df.hdd_60.mean(), 2) == 11.22
    assert round(df.hdd_61.mean(), 2) == 11.79
    assert round(df.cdd_65.mean(), 2) == 4.11
    assert round(df.cdd_66.mean(), 2) == 3.76
    assert round(df.n_hours_kept.mean(), 2) == 729.23
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_compute_temperature_features_billing_monthly_hourly_degree_days(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert df.shape == (27, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 11.8
    assert round(df.hdd_61.mean(), 2) == 12.38
    assert round(df.cdd_65.mean(), 2) == 3.96
    assert round(df.cdd_66.mean(), 2) == 3.62
    assert round(df.n_hours_kept.mean(), 2) == 719.15
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_billing_monthly_hourly_degree_days_use_mean_false(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
        use_mean_daily_values=False,
    )
    assert df.shape == (27, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.meter_value.sum()) == round(meter_data.value.sum()) == 21290.0
    assert round(df.hdd_60.mean(), 2) == 336.54
    assert round(df.hdd_61.mean(), 2) == 353.64
    assert round(df.cdd_65.mean(), 2) == 125.96
    assert round(df.cdd_66.mean(), 2) == 115.09
    assert round(df.n_hours_kept.mean(), 2) == 729.23
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_compute_temperature_features_billing_monthly_hourly_degree_days_use_mean_false(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
        use_mean_daily_values=False,
    )
    assert df.shape == (27, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 343.01
    assert round(df.hdd_61.mean(), 2) == 360.19
    assert round(df.cdd_65.mean(), 2) == 121.29
    assert round(df.cdd_66.mean(), 2) == 110.83
    assert round(df.n_hours_kept.mean(), 2) == 719.15
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_billing_monthly_bad_degree_day_method(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data,
            temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_compute_temperature_features_billing_monthly_bad_degree_day_method(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    with pytest.raises(ValueError):
        compute_temperature_features(
            temperature_data,
            meter_data.index,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_merge_temperature_data_billing_monthly_data_quality(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = merge_temperature_data(
        meter_data, temperature_data, temperature_mean=False, data_quality=True
    )
    assert df.shape == (27, 5)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.meter_value.sum()) == 703.0 != round(meter_data.value.sum())
    assert round(df.temperature_not_null.mean(), 2) == 729.23
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_compute_temperature_features_billing_monthly_data_quality(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data, meter_data.index, temperature_mean=False, data_quality=True
    )
    assert df.shape == (27, 4)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.temperature_not_null.mean(), 2) == 719.15
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_merge_temperature_data_billing_bimonthly_temp_mean(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (14, 4)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.meter_value.sum()) == 352.0 != round(meter_data.value.sum())
    assert round(df.temperature_mean.mean()) == 55.0


def test_compute_temperature_features_billing_bimonthly_temp_mean(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = compute_temperature_features(temperature_data, meter_data.index)
    assert df.shape == (14, 3)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.temperature_mean.mean()) == 53.0


def test_merge_temperature_data_billing_bimonthly_daily_degree_days(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
    )
    assert df.shape == (14, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.meter_value.sum()) == 352.0 != round(meter_data.value.sum())
    assert round(df.hdd_60.mean(), 2) == 10.94
    assert round(df.hdd_61.mean(), 2) == 11.51
    assert round(df.cdd_65.mean(), 2) == 3.65
    assert round(df.cdd_66.mean(), 2) == 3.28
    assert round(df.n_days_kept.mean(), 2) == 61.62
    assert round(df.n_days_dropped.mean(), 2) == 0.0


def test_compute_temperature_features_billing_bimonthly_daily_degree_days(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="daily",
    )
    assert df.shape == (14, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_days_dropped",
        "n_days_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 12.72
    assert round(df.hdd_61.mean(), 2) == 13.32
    assert round(df.cdd_65.mean(), 2) == 3.39
    assert round(df.cdd_66.mean(), 2) == 3.05
    assert round(df.n_days_kept.mean(), 2) == 57.79
    assert round(df.n_days_dropped.mean(), 2) == 0.07


def test_merge_temperature_data_billing_bimonthly_hourly_degree_days(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = merge_temperature_data(
        meter_data,
        temperature_data,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert df.shape == (14, 7)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "meter_value",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.meter_value.sum()) == 352.0 != round(meter_data.value.sum())
    assert round(df.hdd_60.mean(), 2) == 11.33
    assert round(df.hdd_61.mean(), 2) == 11.9
    assert round(df.cdd_65.mean(), 2) == 4.07
    assert round(df.cdd_66.mean(), 2) == 3.72
    assert round(df.n_hours_kept.mean(), 2) == 1478.77
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_compute_temperature_features_billing_bimonthly_hourly_degree_days(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=[60, 61],
        cooling_balance_points=[65, 66],
        temperature_mean=False,
        degree_day_method="hourly",
    )
    assert df.shape == (14, 6)
    assert list(sorted(df.columns)) == [
        "cdd_65",
        "cdd_66",
        "hdd_60",
        "hdd_61",
        "n_hours_dropped",
        "n_hours_kept",
    ]
    assert round(df.hdd_60.mean(), 2) == 13.08
    assert round(df.hdd_61.mean(), 2) == 13.69
    assert round(df.cdd_65.mean(), 2) == 3.78
    assert round(df.cdd_66.mean(), 2) == 3.46
    assert round(df.n_hours_kept.mean(), 2) == 1386.93
    assert round(df.n_hours_dropped.mean(), 2) == 0


def test_merge_temperature_data_billing_bimonthly_bad_degree_days(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    with pytest.raises(ValueError):
        merge_temperature_data(
            meter_data,
            temperature_data,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_compute_temperature_features_billing_bimonthly_bad_degree_days(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    with pytest.raises(ValueError):
        compute_temperature_features(
            temperature_data,
            meter_data.index,
            heating_balance_points=[60, 61],
            cooling_balance_points=[65, 66],
            degree_day_method="UNKNOWN",
        )


def test_merge_temperature_data_billing_bimonthly_data_quality(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = merge_temperature_data(
        meter_data, temperature_data, temperature_mean=False, data_quality=True
    )
    assert df.shape == (14, 5)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.meter_value.sum()) == 352.0 != round(meter_data.value.sum())
    assert round(df.temperature_not_null.mean(), 2) == 1478.77
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_compute_temperature_features_billing_bimonthly_data_quality(
    il_electricity_cdd_hdd_billing_bimonthly
):
    meter_data = il_electricity_cdd_hdd_billing_bimonthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_bimonthly["temperature_data"]
    df = compute_temperature_features(
        temperature_data, meter_data.index, temperature_mean=False, data_quality=True
    )
    assert df.shape == (14, 4)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_not_null",
        "temperature_null",
    ]
    assert round(df.temperature_not_null.mean(), 2) == 1386.93
    assert round(df.temperature_null.mean(), 2) == 0.0


def test_merge_temperature_data_shorter_temperature_data(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]

    # drop some data
    temperature_data = temperature_data[:-200]

    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (810, 4)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.meter_value.sum()) == 21564.0
    assert round(df.temperature_mean.sum()) == 43958.0


def test_compute_temperature_features_shorter_temperature_data(
    il_electricity_cdd_hdd_daily
):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]

    # drop some data
    temperature_data = temperature_data[:-200]

    df = compute_temperature_features(temperature_data, meter_data.index)
    assert df.shape == (810, 3)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.temperature_mean.sum()) == 43958.0


def test_compute_temperature_features_with_duplicated_index(
    il_electricity_cdd_hdd_billing_monthly
):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]

    # these are specifically formed to give a less readable error if
    # duplicates are not caught
    meter_data = meter_data.append(meter_data).sort_index()
    temperature_data = temperature_data.iloc[8000:]

    with pytest.raises(ValueError) as excinfo:
        compute_temperature_features(temperature_data, meter_data.index)
    assert str(excinfo.value) == "Duplicates found in input meter trace index."


def test_merge_temperature_data_shorter_meter_data(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]

    # drop some data
    meter_data = meter_data[:-10]

    df = merge_temperature_data(meter_data, temperature_data)
    assert df.shape == (800, 4)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.meter_value.sum()) == 21525.0
    assert round(df.temperature_mean.sum()) == 43934.0


def test_compute_temperature_features_shorter_meter_data(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]

    # drop some data
    meter_data = meter_data[:-10]

    df = compute_temperature_features(temperature_data, meter_data.index)
    assert df.shape == (800, 3)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.temperature_mean.sum()) == 43934.0


def test_merge_temperature_data_empty_temperature_data():
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="H")
    temperature_data = pd.Series({"value": []}, index=index).astype(float)
    result_index = temperature_data.resample("D").sum().index
    meter_data_hack = pd.DataFrame({"value": 0}, index=result_index)

    df = merge_temperature_data(
        meter_data_hack,
        temperature_data,
        heating_balance_points=[65],
        cooling_balance_points=[65],
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (0, 4)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.meter_value.sum()) == 0
    assert round(df.temperature_mean.sum()) == 0


def test_compute_temperature_features_empty_temperature_data():
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="H")
    temperature_data = pd.Series({"value": []}, index=index).astype(float)
    result_index = temperature_data.resample("D").sum().index
    meter_data_hack = pd.DataFrame({"value": 0}, index=result_index)

    df = compute_temperature_features(
        temperature_data,
        meter_data_hack.index,
        heating_balance_points=[65],
        cooling_balance_points=[65],
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (0, 3)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.temperature_mean.sum()) == 0


def test_merge_temperature_data_empty_meter_data():
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="H")
    temperature_data = pd.Series({"value": 0}, index=index)
    result_index = temperature_data.resample("D").sum().index
    meter_data_hack = pd.DataFrame({"value": []}, index=result_index)
    meter_data_hack.index.freq = None

    df = merge_temperature_data(
        meter_data_hack,
        temperature_data,
        heating_balance_points=[65],
        cooling_balance_points=[65],
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (0, 4)
    assert list(sorted(df.columns)) == [
        "meter_value",
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.meter_value.sum()) == 0
    assert round(df.temperature_mean.sum()) == 0


def test_compute_temperature_features_empty_meter_data():
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="H")
    temperature_data = pd.Series({"value": 0}, index=index)
    result_index = temperature_data.resample("D").sum().index
    meter_data_hack = pd.DataFrame({"value": []}, index=result_index)
    meter_data_hack.index.freq = None

    df = compute_temperature_features(
        temperature_data,
        meter_data_hack.index,
        heating_balance_points=[65],
        cooling_balance_points=[65],
        degree_day_method="daily",
        use_mean_daily_values=False,
    )
    assert df.shape == (0, 3)
    assert list(sorted(df.columns)) == [
        "n_days_dropped",
        "n_days_kept",
        "temperature_mean",
    ]
    assert round(df.temperature_mean.sum()) == 0


def test_as_freq_not_series(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    with pytest.raises(ValueError):
        as_freq(meter_data, freq="H")


def test_as_freq_hourly(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_hourly = as_freq(meter_data.value, freq="H")
    assert as_hourly.shape == (18961,)
    assert round(meter_data.value.sum(), 1) == round(as_hourly.sum(), 1) == 21290.2


def test_as_freq_daily(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_daily = as_freq(meter_data.value, freq="D")
    assert as_daily.shape == (791,)
    assert round(meter_data.value.sum(), 1) == round(as_daily.sum(), 1) == 21290.2


def test_as_freq_month_start(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    assert meter_data.shape == (27, 1)
    as_month_start = as_freq(meter_data.value, freq="MS")
    assert as_month_start.shape == (27,)
    assert round(meter_data.value.sum(), 1) == round(as_month_start.sum(), 1) == 21290.2


def test_as_freq_empty():
    meter_data = pd.DataFrame({"value": []})
    empty_meter_data = as_freq(meter_data.value, freq="H")
    assert empty_meter_data.empty


def test_day_counts(il_electricity_cdd_hdd_billing_monthly):
    data = il_electricity_cdd_hdd_billing_monthly["meter_data"].value
    counts = day_counts(data)
    assert counts.shape == (27,)
    assert counts.iloc[0] == 29.0
    assert pd.isnull(counts.iloc[-1])
    assert counts.sum() == 790.0


def test_day_counts_empty_series():
    index = pd.DatetimeIndex([])
    index.freq = None
    data = pd.Series([], index=index)
    counts = day_counts(data)
    assert counts.shape == (0,)


def test_get_baseline_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    baseline_data, warnings = get_baseline_data(meter_data)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 0


def test_get_baseline_data_with_end(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_data, warnings = get_baseline_data(meter_data, end=blackout_start_date)
    assert meter_data.shape != baseline_data.shape == (8761, 1)
    assert len(warnings) == 0


def test_get_baseline_data_with_end_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date, max_days=None
    )
    assert meter_data.shape != baseline_data.shape == (9595, 1)
    assert len(warnings) == 0


def test_get_baseline_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    with pytest.raises(NoBaselineDataError):
        get_baseline_data(meter_data, end=pd.Timestamp("2000").tz_localize("UTC"))


def test_get_baseline_data_start_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    start = meter_data.index.min() - timedelta(days=1)
    baseline_data, warnings = get_baseline_data(meter_data, start=start)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_baseline_data.gap_at_baseline_start"
    assert (
        warning.description
        == "Data does not have coverage at requested baseline start date."
    )
    assert warning.data == {
        "data_start": "2015-11-22T06:00:00+00:00",
        "requested_start": "2015-11-21T06:00:00+00:00",
    }


def test_get_baseline_data_end_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    end = meter_data.index.max() + timedelta(days=1)
    baseline_data, warnings = get_baseline_data(meter_data, end=end, max_days=None)
    assert meter_data.shape == baseline_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_baseline_data.gap_at_baseline_end"
    assert (
        warning.description
        == "Data does not have coverage at requested baseline end date."
    )
    assert warning.data == {
        "data_end": "2018-02-08T06:00:00+00:00",
        "requested_end": "2018-02-09T06:00:00+00:00",
    }


def test_get_reporting_data(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    reporting_data, warnings = get_reporting_data(meter_data)
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 0


def test_get_reporting_data_with_start(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_data, warnings = get_reporting_data(meter_data, start=blackout_end_date)
    assert meter_data.shape != reporting_data.shape == (8761, 1)
    assert len(warnings) == 0


def test_get_reporting_data_with_start_no_max_days(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=None
    )
    assert meter_data.shape != reporting_data.shape == (9607, 1)
    assert len(warnings) == 0


def test_get_reporting_data_empty(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    with pytest.raises(NoReportingDataError):
        get_reporting_data(meter_data, start=pd.Timestamp("2030").tz_localize("UTC"))


def test_get_reporting_data_start_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    start = meter_data.index.min() - timedelta(days=1)
    reporting_data, warnings = get_reporting_data(
        meter_data, start=start, max_days=None
    )
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_reporting_data.gap_at_reporting_start"
    assert (
        warning.description
        == "Data does not have coverage at requested reporting start date."
    )
    assert warning.data == {
        "data_start": "2015-11-22T06:00:00+00:00",
        "requested_start": "2015-11-21T06:00:00+00:00",
    }


def test_get_reporting_data_end_gap(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    end = meter_data.index.max() + timedelta(days=1)
    reporting_data, warnings = get_reporting_data(meter_data, end=end)
    assert meter_data.shape == reporting_data.shape == (19417, 1)
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == "eemeter.get_reporting_data.gap_at_reporting_end"
    assert (
        warning.description
        == "Data does not have coverage at requested reporting end date."
    )
    assert warning.data == {
        "data_end": "2018-02-08T06:00:00+00:00",
        "requested_end": "2018-02-09T06:00:00+00:00",
    }


def test_remove_duplicates_df():
    index = pd.DatetimeIndex(["2017-01-01", "2017-01-02", "2017-01-02"])
    df = pd.DataFrame({"value": [1, 2, 3]}, index=index)
    assert df.shape == (3, 1)
    df_dedupe = remove_duplicates(df)
    assert df_dedupe.shape == (2, 1)
    assert list(df_dedupe.value) == [1, 2]


def test_remove_duplicates_series():
    index = pd.DatetimeIndex(["2017-01-01", "2017-01-02", "2017-01-02"])
    series = pd.Series([1, 2, 3], index=index)
    assert series.shape == (3,)
    series_dedupe = remove_duplicates(series)
    assert series_dedupe.shape == (2,)
    assert list(series_dedupe) == [1, 2]
