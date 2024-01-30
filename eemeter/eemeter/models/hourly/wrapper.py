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

from eemeter.eemeter.features import estimate_hour_of_week_occupancy, fit_temperature_bins
from eemeter.eemeter.models.design_matrices import create_caltrack_hourly_segmented_design_matrices
from eemeter.eemeter.models.hourly.model import CalTRACKHourlyModelResults, fit_caltrack_hourly_model
from eemeter.eemeter.segmentation import segment_time_series


class HourlyModel:
    def __init__(self, settings=None):
        pass

    def fit(self, preliminary_design_matrix):
        segmentation = segment_time_series(
            preliminary_design_matrix.index, "three_month_weighted"
        )
        occupancy_lookup = estimate_hour_of_week_occupancy(
            preliminary_design_matrix, segmentation=segmentation
        )
        (
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        ) = fit_temperature_bins(
            preliminary_design_matrix,
            segmentation=segmentation,
            occupancy_lookup=occupancy_lookup,
        )
        segmented_design_matrices = create_caltrack_hourly_segmented_design_matrices(
            preliminary_design_matrix,
            segmentation,
            occupancy_lookup,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
        baseline_model = fit_caltrack_hourly_model(
            segmented_design_matrices,
            occupancy_lookup,
            occupied_temperature_bins,
            unoccupied_temperature_bins,
        )
        self.model = baseline_model
        self.is_fitted = True
        return self

    def predict(self, df_eval):
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")
        prediction_index = df_eval.index
        temperature_series = df_eval['temperature_mean']
        model_prediction = self.model.predict(
            prediction_index, temperature_series
        )
        return model_prediction.result

    @classmethod
    def from_dict(cls, data):
        hourly_model = cls()
        hourly_model.model = CalTRACKHourlyModelResults.from_json(data)
        hourly_model.is_fitted = True
        return hourly_model

    @classmethod
    def from_json(cls, str_data):
        return cls.from_dict(json.loads(str_data))

    def plot(
        self,
        ax=None,
        title=None,
        figsize=None,
        temp_range=None,
    ):
        raise NotImplementedError