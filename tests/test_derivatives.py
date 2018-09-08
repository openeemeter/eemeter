import numpy as np
import pandas as pd
import pytest

from eemeter.caltrack import caltrack_method
from eemeter.derivatives import metered_savings, modeled_savings
from eemeter.exceptions import MissingModelParameterError
from eemeter.features import (
    compute_temperature_features,
    compute_usage_per_day_feature,
    merge_features,
)
from eemeter.transform import get_baseline_data


@pytest.fixture
def cdd_hdd_h60_c65(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily["meter_data"]
    temperature_data = il_electricity_cdd_hdd_daily["temperature_data"]
    blackout_start_date = il_electricity_cdd_hdd_daily["blackout_start_date"]
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=[60],
        cooling_balance_points=[65],
        use_mean_daily_values=True,
    )
    meter_data_feature = compute_usage_per_day_feature(meter_data, "meter_value")
    data = merge_features([meter_data_feature, temperature_features])
    baseline_data, warnings = get_baseline_data(data, end=blackout_start_date)
    return baseline_data


@pytest.fixture
def baseline_model(cdd_hdd_h60_c65):
    model_results = caltrack_method(cdd_hdd_h60_c65)
    return model_results.model


@pytest.fixture
def baseline_model_results(cdd_hdd_h60_c65):
    model_results = caltrack_method(cdd_hdd_h60_c65)
    return model_results


@pytest.fixture
def reporting_model(cdd_hdd_h60_c65):
    model_results = caltrack_method(cdd_hdd_h60_c65)
    return model_results.model


@pytest.fixture
def reporting_meter_data():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index)


@pytest.fixture
def reporting_temperature_data():
    index = pd.date_range("2011-01-01", freq="D", periods=60, tz="UTC")
    return pd.Series(np.arange(30.0, 90.0), index=index).asfreq("H").ffill()


def test_metered_savings_cdd_hdd_daily(
    baseline_model_results, reporting_meter_data, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_results,
        reporting_meter_data,
        reporting_temperature_data,
        degree_day_method="daily",
        frequency="daily",
        t_stat=1.649,
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == 1569.57


@pytest.fixture
def reporting_meter_data_billing():
    index = pd.date_range("2011-01-01", freq="MS", periods=13, tz="UTC")
    return pd.DataFrame({"value": 1}, index=index)


def test_metered_savings_cdd_hdd_billing(
    baseline_model_results, reporting_meter_data_billing, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_results,
        reporting_meter_data_billing,
        reporting_temperature_data,
        degree_day_method="daily",
        frequency="billing",
        t_stat=1.649,
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == 1626.57


def test_metered_savings_cdd_hdd_hourly_degree_days(
    baseline_model_results, reporting_meter_data, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_results,
        reporting_meter_data,
        reporting_temperature_data,
        degree_day_method="hourly",
        frequency="daily",
        t_stat=1.649,
    )
    assert list(results.columns) == [
        "reporting_observed",
        "counterfactual_usage",
        "metered_savings",
    ]
    assert round(results.metered_savings.sum(), 2) == 1569.57


def test_metered_savings_cdd_hdd_no_params(
    baseline_model_results, reporting_meter_data, reporting_temperature_data
):
    baseline_model_results.model.model_params = None
    with pytest.raises(MissingModelParameterError):
        metered_savings(
            baseline_model_results,
            reporting_meter_data,
            reporting_temperature_data,
            degree_day_method="daily",
            frequency="daily",
            t_stat=1.649,
        )


def test_metered_savings_cdd_hdd_with_disaggregated(
    baseline_model_results, reporting_meter_data, reporting_temperature_data
):

    results, error_bands = metered_savings(
        baseline_model_results,
        reporting_meter_data,
        reporting_temperature_data,
        degree_day_method="daily",
        with_disaggregated=True,
        frequency="daily",
        t_stat=1.649,
    )
    assert list(sorted(results.columns)) == [
        "counterfactual_base_load",
        "counterfactual_cooling_load",
        "counterfactual_heating_load",
        "counterfactual_usage",
        "metered_savings",
        "reporting_observed",
    ]


def test_modeled_savings_cdd_hdd(
    baseline_model, reporting_model, reporting_meter_data, reporting_temperature_data
):
    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model,
        reporting_model,
        reporting_meter_data.index,
        reporting_temperature_data,
        degree_day_method="daily",
    )
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]
    assert round(results.modeled_savings.sum(), 2) == 0.0


def test_modeled_savings_cdd_hdd_hourly_degree_days(
    baseline_model, reporting_model, reporting_meter_data, reporting_temperature_data
):
    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model,
        reporting_model,
        reporting_meter_data.index,
        reporting_temperature_data,
        degree_day_method="hourly",
    )
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]
    assert round(results.modeled_savings.sum(), 2) == 0.0


def test_modeled_savings_cdd_hdd_baseline_model_no_params(
    baseline_model, reporting_model, reporting_meter_data, reporting_temperature_data
):
    baseline_model.model_params = None
    with pytest.raises(MissingModelParameterError):
        modeled_savings(
            baseline_model,
            reporting_model,
            reporting_meter_data.index,
            reporting_temperature_data,
            degree_day_method="daily",
        )


def test_modeled_savings_cdd_hdd_reporting_model_no_params(
    baseline_model, reporting_model, reporting_meter_data, reporting_temperature_data
):
    reporting_model.model_params = None
    with pytest.raises(MissingModelParameterError):
        modeled_savings(
            baseline_model,
            reporting_model,
            reporting_meter_data.index,
            reporting_temperature_data,
            degree_day_method="daily",
        )


def test_modeled_savings_cdd_hdd_with_disaggregated(
    baseline_model, reporting_model, reporting_meter_data, reporting_temperature_data
):

    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model,
        reporting_model,
        reporting_meter_data.index,
        reporting_temperature_data,
        degree_day_method="daily",
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


def test_modeled_savings_empty_temperature_data(baseline_model, reporting_model):
    index = pd.DatetimeIndex([], tz="UTC", name="dt", freq="H")
    temperature_data = pd.Series([], index=index)

    meter_data_index = temperature_data.resample("D").sum().index

    # using reporting data for convenience, but intention is to use normal data
    results = modeled_savings(
        baseline_model,
        reporting_model,
        meter_data_index,
        temperature_data,
        degree_day_method="daily",
    )
    assert results.shape == (0, 3)
    assert list(results.columns) == [
        "modeled_baseline_usage",
        "modeled_reporting_usage",
        "modeled_savings",
    ]
