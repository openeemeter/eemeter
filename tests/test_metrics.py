import json
import pytest
import pandas as pd
import numpy as np

from eemeter import ModelMetrics
from eemeter.metrics import (
    _compute_r_squared,
    _compute_r_squared_adj,
    _compute_rmse,
    _compute_rmse_adj,
    _compute_cvrmse,
    _compute_cvrmse_adj,
    _compute_mape,
    _compute_nmae,
    _compute_nmbe,
    _compute_autocorr_resid,
    _json_safe_float,
)


@pytest.fixture
def sample_data():
    # Could have included DatetimeIndex, but made it more general
    series_one = pd.Series([1, 3, 4, 1, 6], name="NameOne")
    series_two = pd.Series([2, 3, 3, 2, 4], name="NameTwo")
    return series_one, series_two


def test_ModelMetrics(sample_data):
    series_one, series_two = sample_data
    model_metrics = ModelMetrics(series_one, series_two, num_parameters=2)
    assert model_metrics.observed_length == 5
    assert model_metrics.predicted_length == 5
    assert model_metrics.merged_length == 5
    assert model_metrics.observed_mean == 3.0
    assert model_metrics.predicted_mean == 2.8
    assert round(model_metrics.observed_skew, 3) == 0.524
    assert round(model_metrics.predicted_skew, 3) == 0.512
    assert round(model_metrics.observed_kurtosis, 3) == -0.963
    assert round(model_metrics.predicted_kurtosis, 3) == -0.612
    assert round(model_metrics.observed_cvstd, 3) == 0.707
    assert round(model_metrics.predicted_cvstd, 3) == 0.299
    assert round(model_metrics.r_squared, 3) == 0.972
    assert round(model_metrics.r_squared_adj, 3) == 0.944
    assert round(model_metrics.cvrmse, 3) == 0.394
    assert round(model_metrics.cvrmse_adj, 3) == 0.509
    assert round(model_metrics.mape, 3) == 0.517
    assert round(model_metrics.mape_no_zeros, 3) == 0.517
    assert model_metrics.num_meter_zeros == 0
    assert round(model_metrics.nmae, 3) == 0.333
    assert round(model_metrics.nmbe, 3) == -0.067
    assert round(model_metrics.autocorr_resid, 3) == -0.674
    assert repr(model_metrics) is not None
    assert json.dumps(model_metrics.json()) is not None


@pytest.fixture
def sample_data_zeros():
    series_one = pd.Series([1, 0, 0, 1, 6])
    series_two = pd.Series([2, 3, 3, 2, 4])
    return series_one, series_two


def test_ModelMetrics_zeros(sample_data_zeros):
    series_one, series_two = sample_data_zeros
    model_metrics = ModelMetrics(series_one, series_two, num_parameters=2)
    assert np.isinf(model_metrics.mape)
    assert model_metrics.num_meter_zeros == 2


def test_ModelMetrics_num_parameter_error(sample_data):
    series_one, series_two = sample_data
    with pytest.raises(ValueError):
        model_metrics = ModelMetrics(series_one, series_two, num_parameters=-1)


def test_ModelMetrics_autocorr_lags_error(sample_data):
    series_one, series_two = sample_data
    with pytest.raises(ValueError):
        model_metrics = ModelMetrics(series_one, series_two, autocorr_lags=0)


@pytest.fixture
def sample_data_diff_length_no_nan():
    series_one = pd.Series([1, 0, 0, 1, 6, 4, 5])
    series_two = pd.Series([2, 3, 3, 2, 4])
    return series_one, series_two


def test_ModelMetrics_diff_length_error_no_nan(sample_data_diff_length_no_nan):
    series_one, series_two = sample_data_diff_length_no_nan
    model_metrics = ModelMetrics(series_one, series_two)
    assert len(model_metrics.warnings) == 1
    warning = model_metrics.warnings[0]
    assert warning.qualified_name.startswith("eemeter.metrics.input_series_are_of")
    assert warning.description.startswith("Input series")
    assert warning.data == {
        "merged_length": 5,
        "observed_input_length": 7,
        "observed_length_without_nan": 7,
        "predicted_input_length": 5,
        "predicted_length_without_nan": 5,
    }


@pytest.fixture
def sample_data_diff_length_with_nan():
    series_one = pd.Series([1, 0, 0, 1, 6, 4, 5])
    series_two = pd.Series([2, 3, 3, 2, 4, np.nan, np.nan])
    return series_one, series_two


def test_ModelMetrics_diff_length_error_with_nan(sample_data_diff_length_with_nan):
    series_one, series_two = sample_data_diff_length_with_nan
    model_metrics = ModelMetrics(series_one, series_two)
    assert len(model_metrics.warnings) == 1
    warning = model_metrics.warnings[0]
    assert warning.qualified_name.startswith("eemeter.metrics.input_series_are_of")
    assert warning.description.startswith("Input series")
    assert warning.data == {
        "merged_length": 5,
        "observed_input_length": 7,
        "observed_length_without_nan": 7,
        "predicted_input_length": 7,
        "predicted_length_without_nan": 5,
    }


def test_ModelMetrics_inputs_unchanged(sample_data):
    series_one, series_two = sample_data
    model_metrics = ModelMetrics(series_one, series_two)
    assert sample_data[0].name == "NameOne"
    assert sample_data[1].name == "NameTwo"


@pytest.fixture
def model_metrics(sample_data):
    series_one, series_two = sample_data
    return ModelMetrics(series_one, series_two, num_parameters=2)


def test_model_metrics_json_valid(model_metrics):
    model_metrics.r_squared = np.nan
    model_metrics.r_squared_adj = float("nan")
    model_metrics.cvrmse = np.inf
    model_metrics.cvrmse_adj = float("inf")
    model_metrics.nmae = None
    model_metrics.mape = float("-inf")
    json_rep = model_metrics.json()
    json.dumps(json_rep)
    assert sorted(json_rep.keys()) == [
        "autocorr_resid",
        "cvrmse",
        "cvrmse_adj",
        "mape",
        "mape_no_zeros",
        "merged_length",
        "nmae",
        "nmbe",
        "num_meter_zeros",
        "num_parameters",
        "observed_cvstd",
        "observed_kurtosis",
        "observed_length",
        "observed_mean",
        "observed_skew",
        "observed_variance",
        "predicted_cvstd",
        "predicted_kurtosis",
        "predicted_length",
        "predicted_mean",
        "predicted_skew",
        "predicted_variance",
        "r_squared",
        "r_squared_adj",
        "rmse",
        "rmse_adj",
    ]


@pytest.fixture
def sample_data_merged(sample_data):
    series_one, series_two = sample_data
    observed = series_one.to_frame().dropna()
    predicted = series_two.to_frame().dropna()
    observed.columns = ["observed"]
    predicted.columns = ["predicted"]
    combined = observed.merge(predicted, left_index=True, right_index=True)
    combined["residuals"] = combined.predicted - combined.observed
    return combined


def test_compute_r_squared(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_r_squared(combined), 3) == 0.972


def test_compute_r_squared_adj(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_r_squared_adj(_compute_r_squared(combined), 5, 2), 3) == 0.944


def test_compute_cvrmse(sample_data_merged):
    combined = sample_data_merged
    observed_mean = combined["observed"].mean()
    assert round(_compute_cvrmse(_compute_rmse(combined), observed_mean), 3) == 0.394


def test_compute_cvrmse_adj(sample_data_merged):
    combined = sample_data_merged
    observed_mean = combined["observed"].mean()
    observed_length = len(combined["observed"])
    num_parameters = 2
    rmse_adj = _compute_rmse_adj(combined, observed_length, num_parameters)
    assert round(_compute_cvrmse_adj(rmse_adj, observed_mean), 3) == 0.509


def test_compute_mape(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_mape(combined), 3) == 0.517


def test_compute_nmae(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_nmae(combined), 3) == 0.333


def test_compute_nmbe(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_nmbe(combined), 3) == -0.067


def test_compute_autocorr_resid(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_autocorr_resid(combined, 1), 3) == -0.674


def test_json_safe_float():
    assert _json_safe_float(float("inf")) is None
    assert _json_safe_float(float("-inf")) is None
    assert _json_safe_float(float("nan")) == None
    assert _json_safe_float(np.inf) is None
    assert _json_safe_float(-np.inf) is None
    assert _json_safe_float(np.nan) is None
    assert _json_safe_float(3.3) == 3.3
    assert _json_safe_float("3.3") == 3.3
    assert _json_safe_float(1) == 1.0
    assert _json_safe_float(None) == None

    with pytest.raises(Exception):
        _json_safe_float("not a number")
