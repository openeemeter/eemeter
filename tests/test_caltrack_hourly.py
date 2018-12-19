#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2018 Open Energy Efficiency, Inc.

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


@pytest.fixture
def segmented_data():
    index = pd.date_range(start="2017-01-01", periods=24, freq="H", tz="UTC")
    time_features = compute_time_features(index)
    segmented_data = pd.DataFrame(
        {
            "hour_of_week": time_features.hour_of_week,
            "temperature_mean": np.linspace(0, 100, 24),
            "meter_value": np.linspace(10, 70, 24),
            "weight": np.ones((24,)),
        },
        index=index,
    )
    return segmented_data


@pytest.fixture
def occupancy_lookup():
    index = pd.Categorical(range(168))
    occupancy = pd.Series([i % 2 == 0 for i in range(168)], index=index)
    return pd.DataFrame(
        {"dec-jan-feb-weighted": occupancy, "jan-feb-mar-weighted": occupancy}
    )


@pytest.fixture
def temperature_bins():
    bins = pd.Series([True, True, True], index=[30, 60, 90])
    return pd.DataFrame({"dec-jan-feb-weighted": bins, "jan-feb-mar-weighted": bins})


def test_caltrack_hourly_fit_feature_processor(
    segmented_data, occupancy_lookup, temperature_bins
):
    result = caltrack_hourly_fit_feature_processor(
        "dec-jan-feb-weighted", segmented_data, occupancy_lookup, temperature_bins
    )
    assert list(result.columns) == [
        "meter_value",
        "hour_of_week",
        "occupancy",
        "bin_0",
        "bin_1",
        "bin_2",
        "bin_3",
        "weight",
    ]
    assert result.shape == (24, 8)
    assert round(result.sum().sum(), 2) == 5928.0


def test_caltrack_hourly_prediction_feature_processor(
    segmented_data, occupancy_lookup, temperature_bins
):
    result = caltrack_hourly_prediction_feature_processor(
        "dec-jan-feb-weighted", segmented_data, occupancy_lookup, temperature_bins
    )
    assert list(result.columns) == [
        "hour_of_week",
        "occupancy",
        "bin_0",
        "bin_1",
        "bin_2",
        "bin_3",
        "weight",
    ]
    assert result.shape == (24, 7)
    assert round(result.sum().sum(), 2) == 4968.0


@pytest.fixture
def segmented_design_matrices(segmented_data, occupancy_lookup, temperature_bins):
    return {
        "dec-jan-feb-weighted": caltrack_hourly_fit_feature_processor(
            "dec-jan-feb-weighted", segmented_data, occupancy_lookup, temperature_bins
        )
    }


def test_fit_caltrack_hourly_model_segment(segmented_design_matrices):
    segment_name = "dec-jan-feb-weighted"
    segment_data = segmented_design_matrices[segment_name]
    segment_model = fit_caltrack_hourly_model_segment(segment_name, segment_data)
    assert segment_model.formula == (
        "meter_value ~ C(hour_of_week) - 1 + bin_0:C(occupancy)"
        " + bin_1:C(occupancy) + bin_2:C(occupancy) + bin_3:C(occupancy)"
    )
    assert segment_model.segment_name == "dec-jan-feb-weighted"
    assert len(segment_model.model_params.keys()) == 32
    assert segment_model.model is not None
    assert segment_model.warnings is not None
    prediction = segment_model.predict(segment_data)
    assert round(prediction.sum(), 2) == 960.0


@pytest.fixture
def temps():
    index = pd.date_range(start="2017-01-01", periods=24, freq="H", tz="UTC")
    temps = pd.Series(np.linspace(0, 100, 24), index=index)
    return temps


def test_fit_caltrack_hourly_model(
    segmented_design_matrices, occupancy_lookup, temperature_bins, temps
):
    segmented_model = fit_caltrack_hourly_model(
        segmented_design_matrices, occupancy_lookup, temperature_bins
    )

    assert segmented_model.segment_models is not None
    prediction = segmented_model.predict(temps.index, temps).result


def test_serialize_caltrack_hourly_model(
    segmented_design_matrices, occupancy_lookup, temperature_bins, temps
):
    segmented_model = fit_caltrack_hourly_model(
        segmented_design_matrices, occupancy_lookup, temperature_bins
    )
    assert json.dumps(segmented_model.json())
