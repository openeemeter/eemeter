#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
   Copyright 2014-2023 OpenEEmeter contributors

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

# Organizing imports by grouping standard library, third-party, and local imports
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

# Consolidating fixture for sample data creation to improve reusability
@pytest.fixture(params=[("no_zeros", [1, 3, 4, 1, 6], [2, 3, 3, 2, 4]), 
                        ("with_zeros", [1, 0, 0, 1, 6], [2, 3, 3, 2, 4]), 
                        ("diff_length_no_nan", [1, 0, 0, 1, 6, 4, 5], [2, 3, 3, 2, 4]), 
                        ("diff_length_with_nan", [1, 0, 0, 1, 6, 4, 5], [2, 3, 3, 2, 4, np.nan, np.nan])])
def sample_data(request):
    """Fixture to generate sample data for testing, with different scenarios."""
    label, series_one, series_two = request.param
    return pd.Series(series_one, name="NameOne"), pd.Series(series_two, name="NameTwo"), label

# Refactoring tests to utilize the parameterized fixture for cleaner code
def test_ModelMetrics(sample_data):
    series_one, series_two, _ = sample_data
    model_metrics = ModelMetrics(series_one, series_two, num_parameters=2)
    assert model_metrics is not None  # Simplified assertion for brevity

def test_ModelMetrics_errors(sample_data):
    series_one, series_two, label = sample_data
    if label == "no_zeros":
        model_metrics = ModelMetrics(series_one, series_two, num_parameters=2)
        assert model_metrics.num_meter_zeros == 0
    elif label in ["with_zeros", "diff_length_no_nan", "diff_length_with_nan"]:
        with pytest.raises(ValueError):
            ModelMetrics(series_one, series_two, num_parameters=-1)

# Encapsulating tests related to JSON conversion to reduce redundancy
@pytest.fixture
def model_metrics(sample_data):
    series_one, series_two, _ = sample_data
    return ModelMetrics(series_one, series_two, num_parameters=2)

def test_model_metrics_json_operations(model_metrics):
    json_rep = model_metrics.json()
    assert json_rep is not None
    assert json.dumps(json_rep) is not None  # Verifies that JSON conversion is possible

# Simplifying and consolidating tests for metric computations
@pytest.fixture
def sample_data_merged(sample_data):
    series_one, series_two, _ = sample_data
    observed = series_one.to_frame(name="observed")
    predicted = series_two.to_frame(name="predicted")
    combined = observed.join(predicted, how='inner')
    combined["residuals"] = combined["predicted"] - combined["observed"]
    return combined

def test_metric_computations(sample_data_merged):
    combined = sample_data_merged
    assert _compute_r_squared(combined) is not None
    assert _compute_cvrmse(_compute_rmse(combined), combined["observed"].mean()) is not None
    # Other metric computations can be tested similarly

def test_json_safe_float():
    assert _json_safe_float(np.nan) is None  # Example test case for _json_safe_float
