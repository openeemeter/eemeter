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
import pandas as pd

from eemeter.features import (
    compute_time_features,
    compute_temperature_features,
    compute_usage_per_day_feature,
    merge_features,
)
from eemeter.segmentation import iterate_segmented_dataset
from eemeter.caltrack.hourly import caltrack_hourly_fit_feature_processor


__all__ = (
    "create_caltrack_hourly_preliminary_design_matrix",
    "create_caltrack_hourly_segmented_design_matrices",
    "create_caltrack_daily_design_matrix",
    "create_caltrack_billing_design_matrix",
)


def create_caltrack_hourly_preliminary_design_matrix(meter_data, temperature_data):
    time_features = compute_time_features(
        meter_data.index, hour_of_week=True, hour_of_day=False, day_of_week=False
    )
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=[50],
        cooling_balance_points=[65],
        degree_day_method="hourly",
    )
    design_matrix = merge_features(
        [meter_data.value.to_frame("meter_value"), temperature_features, time_features]
    )
    return design_matrix


def create_caltrack_billing_design_matrix(meter_data, temperature_data):
    usage_per_day = compute_usage_per_day_feature(meter_data, series_name="meter_value")
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=range(30, 91),
        cooling_balance_points=range(30, 91),
        data_quality=True,
        tolerance=pd.Timedelta(
            "35D"
        ),  # limit temperature data matching to periods of up to 35 days.
    )
    design_matrix = merge_features([usage_per_day, temperature_features])
    return design_matrix


def create_caltrack_daily_design_matrix(meter_data, temperature_data):
    usage_per_day = compute_usage_per_day_feature(meter_data, series_name="meter_value")
    temperature_features = compute_temperature_features(
        meter_data.index,
        temperature_data,
        heating_balance_points=range(30, 91),
        cooling_balance_points=range(30, 91),
        data_quality=True,
    )
    design_matrix = merge_features([usage_per_day, temperature_features])
    return design_matrix


def create_caltrack_hourly_segmented_design_matrices(
    preliminary_design_matrix, segmentation, occupancy_lookup, temperature_bins
):
    return {
        segment_name: segmented_data
        for segment_name, segmented_data in iterate_segmented_dataset(
            preliminary_design_matrix,
            segmentation=segmentation,
            feature_processor=caltrack_hourly_fit_feature_processor,
            feature_processor_kwargs={
                "occupancy_lookup": occupancy_lookup,
                "temperature_bins": temperature_bins,
            },
        )
    }
