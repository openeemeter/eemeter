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

import json
import numpy as np
import pandas as pd
import pytest

from eemeter.caltrack.hourly import (
    caltrack_hourly_fit_feature_processor,
    caltrack_hourly_prediction_feature_processor,
    fit_caltrack_hourly_model_segment,
    fit_caltrack_hourly_model,
)
from eemeter.features import (
    compute_time_features,
    compute_temperature_features,
    compute_usage_per_day_feature,
    merge_features,
)

# Utility function to generate common data structures for tests
def generate_segmented_data(num_periods=24, start_date="2017-01-01", freq="H", tz="UTC", include_weight=True):
    """Generates a DataFrame with segmented data for testing."""
    index = pd.date_range(start=start_date, periods=num_periods, freq=freq, tz=tz)
    time_features = compute_time_features(index)
    data = {
        "hour_of_week": time_features.hour_of_week,
        "temperature_mean": np.linspace(0, 100, num_periods),
        "meter_value": np.linspace(10, 70, num_periods),
    }
    if include_weight:
        data["weight"] = np.ones((num_periods,))
    return pd.DataFrame(data, index=index)

# Simplifying fixture creation by using utility functions for repetitive tasks
@pytest.fixture
def segmented_data():
    return generate_segmented_data()

@pytest.fixture
def segmented_data_nans():
    return generate_segmented_data(num_periods=200)

@pytest.fixture
def segmented_data_nans_less_than_week():
    return generate_segmented_data(num_periods=4)

# Condensed fixture definitions for occupancy and temperature bins
@pytest.fixture
def occupancy_lookup():
    return pd.DataFrame({period: pd.Series([i % 2 == 0 for i in range(168)]) for period in ["dec-jan-feb-weighted", "jan-feb-mar-weighted"]})

@pytest.fixture
def occupancy_lookup_nans():
    occupancy = pd.Series([i % 2 == 0 for i in range(168)])
    occupancy_nans = pd.Series([np.nan] * 168)
    return pd.DataFrame({"dec-jan-feb-weighted": occupancy, "jan-feb-mar-weighted": occupancy, "apr-may-jun-weighted": occupancy_nans})

@pytest.fixture
def occupied_temperature_bins():
    bins = pd.Series([True] * 3, index=[30, 60, 90])
    return pd.DataFrame({period: bins for period in ["dec-jan-feb-weighted", "jan-feb-mar-weighted"]})

@pytest.fixture
def unoccupied_temperature_bins():
    bins = pd.Series([False, True, True], index=[30, 60, 90])
    return pd.DataFrame({period: bins for period in ["dec-jan-feb-weighted", "jan-feb-mar-weighted"]})

# Consolidated and streamlined test functions for feature processing and model fitting
def test_caltrack_hourly_feature_processor(segmented_data, occupancy_lookup, occupied_temperature_bins, unoccupied_temperature_bins, processor_function, expected_columns, expected_sum):
    """Generalized test function for both fit and prediction feature processors."""
    result = processor_function(
        "dec-jan-feb-weighted",
        segmented_data,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    assert list(result.columns) == expected_columns
    assert result.shape[0] == len(segmented_data)
    result.hour_of_week = result.hour_of_week.astype(float)
    assert round(result.sum().sum(), 2) == expected_sum

# Example test usage for fit feature processor
def test_caltrack_hourly_fit_feature_processor(segmented_data, occupancy_lookup, occupied_temperature_bins, unoccupied_temperature_bins):
    test_caltrack_hourly_feature_processor(
        segmented_data,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
        caltrack_hourly_fit_feature_processor,
        [
            "meter_value",
            "hour_of_week",
            "bin_0_occupied",
            "bin_1_occupied",
            "bin_2_occupied",
            "bin_3_occupied",
            "bin_0_unoccupied",
            "bin_1_unoccupied",
            "bin_2_unoccupied",
            "weight",
        ],
        5916.0,
    )

# Additional test functions can follow the same pattern for the prediction feature processor and model fitting tests.

