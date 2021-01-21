#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2019 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
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

from eemeter.caltrack.usage_per_day import fit_caltrack_usage_per_day_model


@pytest.fixture
def sample_data():
    # Could have included DatetimeIndex, but made it more general
    series_one = pd.Series([1, 3, 4, 1, 6], name="NameOne")
    series_two = pd.Series([2, 3, 3, 2, 4], name="NameTwo")
    return series_one, series_two


def test_sample_model_metrics(model_metrics):
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
    assert round(model_metrics.n_prime, 3) == 25.694
    assert round(model_metrics.single_tailed_confidence_level, 3) == 0.95
    assert round(model_metrics.degrees_of_freedom, 3) == 24
    assert round(model_metrics.t_stat, 3) == 1.711
    assert round(model_metrics.cvrmse_auto_corr_correction, 3) == 0.356
    assert round(model_metrics.approx_factor_auto_corr_correction, 3) == 1.038


def test_ModelMetrics(sample_data):
    series_one, series_two = sample_data
    model_metrics = ModelMetrics(series_one, series_two, num_parameters=2)
    test_sample_model_metrics(model_metrics)
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


def test_ModelMetrics_invalid_confidence_level(sample_data):
    series_one, series_two = sample_data
    with pytest.raises(Exception) as e:
        model_metrics = ModelMetrics(
            series_one, series_two, num_parameters=2, confidence_level=1.1
        )

    with pytest.raises(Exception) as e:
        model_metrics = ModelMetrics(
            series_one, series_two, num_parameters=2, confidence_level=-1
        )


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
        "approx_factor_auto_corr_correction",
        "autocorr_resid",
        "confidence_level",
        "cvrmse",
        "cvrmse_adj",
        "cvrmse_auto_corr_correction",
        "degrees_of_freedom",
        "fsu_base_term",
        "mape",
        "mape_no_zeros",
        "merged_length",
        "n_prime",
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
        "single_tailed_confidence_level",
        "t_stat",
    ]


def test_model_metrics_json_covert(sample_data):
    series_one, series_two = sample_data
    model_metrics = ModelMetrics(series_one, series_two, num_parameters=2)
    json_rep = model_metrics.json()
    test_sample_model_metrics(ModelMetrics.from_json(json_rep))


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


def test_total_average_metrics():
    data = pd.DataFrame(
        {
            "meter_value": [6, 1, 1, 6],
            "cdd_65": [5, 0, 0.1, 0],
            "hdd_65": [0, 0.1, 0.1, 5],
            "start": pd.date_range(start="2016-01-02", periods=4, freq="D", tz="UTC"),
        }
    ).set_index("start")

    model_results = fit_caltrack_usage_per_day_model(data, fit_intercept_only=True)
    json_result = model_results.json()
    totals_metrics = json_result["totals_metrics"]
    assert round(totals_metrics["observed_length"], 3) == 3.000
    assert round(totals_metrics["predicted_length"], 3) == 3.000
    assert round(totals_metrics["merged_length"], 3) == 3.000
    assert round(totals_metrics["num_parameters"], 3) == 0
    assert round(totals_metrics["observed_mean"], 3) == 2.667
    assert round(totals_metrics["predicted_mean"], 3) == 3.5
    assert round(totals_metrics["observed_variance"], 3) == 5.556
    assert round(totals_metrics["predicted_variance"], 3) == 0
    assert round(totals_metrics["observed_skew"], 3) == 1.732
    assert round(totals_metrics["predicted_skew"], 3) == 0
    assert round(totals_metrics["observed_cvstd"], 3) == 1.083
    assert round(totals_metrics["predicted_cvstd"], 3) == 0
    assert round(totals_metrics["rmse"], 3) == 2.5
    assert round(totals_metrics["rmse_adj"], 3) == 2.5
    assert round(totals_metrics["cvrmse"], 3) == 0.938
    assert round(totals_metrics["cvrmse_adj"], 3) == 0.938
    assert round(totals_metrics["mape"], 3) == 1.806
    assert round(totals_metrics["mape_no_zeros"], 3) == 1.806
    assert round(totals_metrics["num_meter_zeros"], 3) == 0
    assert round(totals_metrics["nmae"], 3) == 0.938
    assert round(totals_metrics["nmbe"], 3) == 0.312
    assert round(totals_metrics["confidence_level"], 3) == 0.9
    assert round(totals_metrics["single_tailed_confidence_level"], 3) == 0.95

    assert totals_metrics["observed_kurtosis"] is None
    assert totals_metrics["predicted_kurtosis"] is None
    assert totals_metrics["r_squared"] is None
    assert totals_metrics["r_squared_adj"] is None
    assert totals_metrics["autocorr_resid"] is None
    assert totals_metrics["n_prime"] is None
    assert totals_metrics["degrees_of_freedom"] is None
    assert totals_metrics["t_stat"] is None
    assert totals_metrics["cvrmse_auto_corr_correction"] is None
    assert totals_metrics["approx_factor_auto_corr_correction"] is None
    assert totals_metrics["fsu_base_term"] is None

    json_result = model_results.json()
    avgs_metrics = json_result["avgs_metrics"]
    assert round(avgs_metrics["observed_length"], 3) == 4.000
    assert round(avgs_metrics["predicted_length"], 3) == 4.000
    assert round(avgs_metrics["merged_length"], 3) == 4.000
    assert round(avgs_metrics["num_parameters"], 3) == 0
    assert round(avgs_metrics["observed_mean"], 3) == 3.5
    assert round(avgs_metrics["predicted_mean"], 3) == 3.5
    assert round(avgs_metrics["observed_variance"], 3) == 6.25
    assert round(avgs_metrics["predicted_variance"], 3) == 0
    assert round(avgs_metrics["observed_skew"], 3) == 0
    assert round(avgs_metrics["predicted_skew"], 3) == 0
    assert round(avgs_metrics["observed_cvstd"], 3) == 0.825
    assert round(avgs_metrics["predicted_cvstd"], 3) == 0
    assert round(avgs_metrics["observed_kurtosis"], 3) == -6.0
    assert round(avgs_metrics["predicted_kurtosis"], 3) == 0

    assert round(avgs_metrics["rmse"], 3) == 2.5
    assert round(avgs_metrics["rmse_adj"], 3) == 2.5
    assert round(avgs_metrics["cvrmse"], 3) == 0.714
    assert round(avgs_metrics["cvrmse_adj"], 3) == 0.714
    assert round(avgs_metrics["mape"], 3) == 1.458
    assert round(avgs_metrics["mape_no_zeros"], 3) == 1.458
    assert round(avgs_metrics["num_meter_zeros"], 3) == 0
    assert round(avgs_metrics["nmae"], 3) == 0.714
    assert round(avgs_metrics["nmbe"], 3) == 0
    assert round(avgs_metrics["confidence_level"], 3) == 0.9
    assert round(avgs_metrics["n_prime"], 3) == 12.0
    assert round(avgs_metrics["single_tailed_confidence_level"], 3) == 0.95
    assert round(avgs_metrics["autocorr_resid"], 3) == -0.5
    assert round(avgs_metrics["degrees_of_freedom"], 3) == 12.0
    assert round(avgs_metrics["t_stat"], 3) == 1.782
    assert round(avgs_metrics["cvrmse_auto_corr_correction"], 3) == 0.577
    assert round(avgs_metrics["approx_factor_auto_corr_correction"], 3) == 1.08
    assert round(avgs_metrics["fsu_base_term"], 3) == 0.794

    assert avgs_metrics["r_squared"] is None
    assert avgs_metrics["r_squared_adj"] is None


#  'avgs_metrics': {'observed_length': 4.0,
#   'predicted_length': 4.0,
#   'merged_length': 4.0,
#   'num_parameters': 0.0,
#   'observed_mean': 3.5,
#   'predicted_mean': 3.5,
#   'observed_variance': 6.25,
#   'predicted_variance': 0.0,
#   'observed_skew': 0.0,
#   'predicted_skew': 0.0,
#   'observed_kurtosis': -6.0,
#   'predicted_kurtosis': 0.0,
#   'observed_cvstd': 0.8247860988423226,
#   'predicted_cvstd': 0.0,
#   'r_squared': None,
#   'r_squared_adj': None,
#   'rmse': 2.5,
#   'rmse_adj': 2.5,
#   'cvrmse': 0.7142857142857143,
#   'cvrmse_adj': 0.7142857142857143,
#   'mape': 1.4583333333333333,
#   'mape_no_zeros': 1.4583333333333333,
#   'num_meter_zeros': 0.0,
#   'nmae': 0.7142857142857143,
#   'nmbe': 0.0,
#   'autocorr_resid': -0.49999999999999994,
#   'confidence_level': 0.9,
#   'n_prime': 12.0,
#   'single_tailed_confidence_level': 0.95,
#   'degrees_of_freedom': 12.0,
#   't_stat': 1.782287555649159,
#   'cvrmse_auto_corr_correction': 0.5773502691896257,
#   'approx_factor_auto_corr_correction': 1.0801234497346435,
#   'fsu_base_term': 0.7938939759464224},
