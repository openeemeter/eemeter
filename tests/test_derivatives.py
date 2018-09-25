import numpy as np
import pandas as pd
import pytest

from eemeter.caltrack.design_matrices import (
    create_caltrack_daily_design_matrix,
    create_caltrack_billing_design_matrix,
    create_caltrack_hourly_preliminary_design_matrix,
    create_caltrack_hourly_segmented_design_matrices,
)
from eemeter.caltrack.hourly import fit_caltrack_hourly_model
from eemeter.caltrack.usage_per_day import fit_caltrack_usage_per_day_model
from eemeter.derivatives import metered_savings, modeled_savings
from eemeter.exceptions import MissingModelParameterError
from eemeter.features import estimate_hour_of_week_occupancy, fit_temperature_bins
from eemeter.segmentation import segment_time_series
from eemeter.transform import get_baseline_data, get_reporting_data


@pytest.fixture
def baseline_data_daily(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_daily["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    baseline_data = create_caltrack_daily_design_matrix(
        baseline_meter_data, temperature_data
    )
    return baseline_data


@pytest.fixture
def baseline_model_daily(baseline_data_daily):
    model_results = fit_caltrack_usage_per_day_model(baseline_data_daily)
    return model_results


@pytest.fixture
def reporting_data_daily(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    blackout_end_date = il_electricity_cdd_hdd_daily["blackout_end_date"]
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date
    )
    reporting_data = create_caltrack_daily_design_matrix(
        reporting_meter_data, temperature_data
    )
    return reporting_data


@pytest.fixture
def reporting_model_daily(reporting_data_daily):
    model_results = fit_caltrack_usage_per_day_model(reporting_data_daily)
    return model_results


@pytest.fixture
def reporting_meter_data_daily():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index)


@pytest.fixture
def reporting_temperature_data():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.Series(np.arange(30.0, 90.0), index=index).asfreq("H").ffill()


def test_metered_savings_cdd_hdd_daily(
    baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == 1571.28
    assert sorted(error_bands.keys()) == [
        "FSU Error Band",
        "OLS Error Band",
        "OLS Error Band: Model Error",
        "OLS Error Band: Noise",
    ]


@pytest.fixture
def baseline_model_billing(il_electricity_cdd_hdd_billing_monthly):
    meter_data = il_electricity_cdd_hdd_billing_monthly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_billing_monthly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_billing_monthly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    baseline_data = create_caltrack_billing_design_matrix(
        baseline_meter_data, temperature_data
    )
    model_results = fit_caltrack_usage_per_day_model(
        baseline_data, use_billing_presets=True, weights_col="n_days_kept"
    )
    return model_results


@pytest.fixture
def reporting_meter_data_billing():
    index = pd.date_range("2011-01-01", freq="MS", periods=13, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index)


def test_metered_savings_cdd_hdd_billing(
    baseline_model_billing, reporting_meter_data_billing, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_billing, reporting_meter_data_billing, reporting_temperature_data
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == 1625.73
    assert sorted(error_bands.keys()) == [
        "FSU Error Band",
        "OLS Error Band",
        "OLS Error Band: Model Error",
        "OLS Error Band: Noise",
    ]


def test_metered_savings_cdd_hdd_daily_hourly_degree_days(
    baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == 1571.28


def test_metered_savings_cdd_hdd_no_params(
    baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
):
    baseline_model_daily.model.model_params = None
    with pytest.raises(MissingModelParameterError):
        metered_savings(
            baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
        )


def test_metered_savings_cdd_hdd_daily_with_disaggregated(
    baseline_model_daily, reporting_meter_data_daily, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_daily,
        reporting_meter_data_daily,
        reporting_temperature_data,
        with_disaggregated=True,
    )
    assert list(sorted(results.columns)) == [
        "counterfactual_base_load",
        "counterfactual_cooling_load",
        "counterfactual_heating_load",
        "counterfactual_usage",
        "metered_savings",
        "reporting_observed",
    ]


def test_modeled_savings_cdd_hdd_daily(
    baseline_model_daily,
    reporting_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
):
    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model_daily,
        reporting_model_daily,
        reporting_meter_data_daily.index,
        reporting_temperature_data,
    )
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]
    assert round(results.modeled_savings.sum(), 2) == 168.58


def test_modeled_savings_cdd_hdd_daily_hourly_degree_days(
    baseline_model_daily,
    reporting_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
):
    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model_daily,
        reporting_model_daily,
        reporting_meter_data_daily.index,
        reporting_temperature_data,
        predict_kwargs={"degree_day_method": "hourly"},
    )
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]
    assert round(results.modeled_savings.sum(), 2) == 168.58


def test_modeled_savings_cdd_hdd_daily_baseline_model_no_params(
    baseline_model_daily,
    reporting_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
):
    baseline_model_daily.model.model_params = None
    with pytest.raises(MissingModelParameterError):
        modeled_savings(
            baseline_model_daily,
            reporting_model_daily,
            reporting_meter_data_daily.index,
            reporting_temperature_data,
        )


def test_modeled_savings_cdd_hdd_daily_reporting_model_no_params(
    baseline_model_daily,
    reporting_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
):
    reporting_model_daily.model.model_params = None
    with pytest.raises(MissingModelParameterError):
        modeled_savings(
            baseline_model_daily,
            reporting_model_daily,
            reporting_meter_data_daily.index,
            reporting_temperature_data,
        )


def test_modeled_savings_cdd_hdd_daily_with_disaggregated(
    baseline_model_daily,
    reporting_model_daily,
    reporting_meter_data_daily,
    reporting_temperature_data,
):

    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model_daily,
        reporting_model_daily,
        reporting_meter_data_daily.index,
        reporting_temperature_data,
        with_disaggregated=True,
    )
    assert list(sorted(results.columns)) == [
        "modeled_base_load_savings",
        "modeled_baseline_base_load",
        "modeled_baseline_cooling_load",
        "modeled_baseline_heating_load",
        "modeled_baseline_usage",
        "modeled_cooling_load_savings",
        "modeled_heating_load_savings",
        "modeled_reporting_base_load",
        "modeled_reporting_cooling_load",
        "modeled_reporting_heating_load",
        "modeled_reporting_usage",
        "modeled_savings",
    ]


def test_modeled_savings_daily_empty_temperature_data(
    baseline_model_daily, reporting_model_daily
):
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="H")
    temperature_data = pd.Series([], index=index)

    meter_data_index = temperature_data.resample("D").sum().index

    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model_daily, reporting_model_daily, meter_data_index, temperature_data
    )
    assert results.shape == (0, 3)
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]


@pytest.fixture
def baseline_model_hourly(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_hourly["blackout_start_date"]
    baseline_meter_data, warnings = get_baseline_data(
        meter_data, end=blackout_start_date
    )
    preliminary_hourly_design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        baseline_meter_data, temperature_data
    )
    segmentation = segment_time_series(
        preliminary_hourly_design_matrix.index, "three_month_weighted"
    )
    occupancy_lookup = estimate_hour_of_week_occupancy(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )
    temperature_bins = fit_temperature_bins(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        temperature_bins,
    )
    segmented_model = fit_caltrack_hourly_model(
        design_matrices, occupancy_lookup, temperature_bins
    )
    return segmented_model


@pytest.fixture
def reporting_model_hourly(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]
    blackout_end_date = il_electricity_cdd_hdd_hourly["blackout_end_date"]
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date
    )
    preliminary_hourly_design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        reporting_meter_data, temperature_data
    )
    segmentation = segment_time_series(
        preliminary_hourly_design_matrix.index, "three_month_weighted"
    )
    occupancy_lookup = estimate_hour_of_week_occupancy(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )
    temperature_bins = fit_temperature_bins(
        preliminary_hourly_design_matrix, segmentation=segmentation
    )
    design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_hourly_design_matrix,
        segmentation,
        occupancy_lookup,
        temperature_bins,
    )
    segmented_model = fit_caltrack_hourly_model(
        design_matrices, occupancy_lookup, temperature_bins
    )
    return segmented_model


@pytest.fixture
def reporting_meter_data_hourly():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index).asfreq("H").ffill()


def test_metered_savings_cdd_hdd_hourly(
    baseline_model_hourly, reporting_meter_data_hourly, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_hourly, reporting_meter_data_hourly, reporting_temperature_data
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == -428.63


def test_modeled_savings_cdd_hdd_hourly(
    baseline_model_hourly,
    reporting_model_hourly,
    reporting_meter_data_hourly,
    reporting_temperature_data,
):
    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model_hourly,
        reporting_model_hourly,
        reporting_meter_data_hourly.index,
        reporting_temperature_data,
    )
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]
    assert round(results.modeled_savings.sum(), 2) == 20.76
