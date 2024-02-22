#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

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
import numpy as np
import pandas as pd
import json

from eemeter.eemeter.common.features import estimate_hour_of_week_occupancy, fit_temperature_bins
from eemeter.eemeter.models.hourly.design_matrices import create_caltrack_hourly_segmented_design_matrices, create_caltrack_hourly_preliminary_design_matrix
from eemeter.eemeter.models.hourly.model import CalTRACKHourlyModelResults, fit_caltrack_hourly_model
from eemeter.eemeter.models.hourly.derivatives import _compute_error_bands_modeled_savings
from eemeter.eemeter.models.hourly.segmentation import segment_time_series

from eemeter.common.utils import unc_factor, t_stat


month_dict = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}


class HourlyModel:
    def __init__(self, settings=None):
        self.segment_type = "three_month_weighted"
        self.alpha = 0.1

    def fit(self, data):
        meter_data = data.df['observed'].to_frame('value')
        temperature_data = data.df['temperature']

        self._preliminary_design_matrix = create_caltrack_hourly_preliminary_design_matrix(meter_data, temperature_data)
        self._segmentation = segment_time_series(
            self._preliminary_design_matrix.index, self.segment_type
        )
        self._occupancy_lookup = estimate_hour_of_week_occupancy(
            self._preliminary_design_matrix, segmentation=self._segmentation
        )
        (
            self._occupied_temperature_bins,
            self._unoccupied_temperature_bins,
        ) = fit_temperature_bins(
            self._preliminary_design_matrix,
            segmentation=self._segmentation,
            occupancy_lookup=self._occupancy_lookup,
        )
        self._segmented_design_matrices = create_caltrack_hourly_segmented_design_matrices(
            self._preliminary_design_matrix,
            self._segmentation,
            self._occupancy_lookup,
            self._occupied_temperature_bins,
            self._unoccupied_temperature_bins,
        )
        self.model = fit_caltrack_hourly_model(
            self._segmented_design_matrices,
            self._occupancy_lookup,
            self._occupied_temperature_bins,
            self._unoccupied_temperature_bins,
            self.segment_type,
        )

        self.is_fitted = True
        self.model_metrics = self.model.totals_metrics

        prediction = self.model.predict(temperature_data.index, temperature_data)
        meter_data = meter_data.merge(prediction.result, left_index=True, right_index=True)
        meter_data.dropna(inplace=True)
        meter_data["resid"] = meter_data["value"] - meter_data["predicted_usage"]

        # get uncertainty variables
        self._autocorr_unc_vars = {}
        if list(self.model_metrics.keys()) == ["all"]:
            self._autocorr_unc_vars["all"] = {
                "mean_baseline_usage": np.mean(meter_data["value"]),
                "n": self.model_metrics["all"].observed_length,
                "n_prime": self.model_metrics["all"].n_prime,
                "MSE": np.mean(meter_data["resid"]**2),
            }
        else:
            # monthly segment model
            model_month_dict = {k.replace("-weighted", "").split("-")[1]:k for k in self.model_metrics.keys()}
            meter_data["month"] = meter_data.index.month

            for month_abbr, model_key in model_month_dict.items():
                month_n = month_dict[month_abbr]
                month_data = meter_data[meter_data["month"] == month_n]

                self._autocorr_unc_vars[month_n] = {
                    "mean_baseline_usage": np.mean(month_data["value"]),
                    "n": self.model_metrics[model_key].observed_length,
                    "n_prime": self.model_metrics[model_key].n_prime,
                    "MSE": np.mean(month_data["resid"]**2),
                }

        return self

    def predict(self, reporting_data):
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")
        prediction_index = reporting_data.df.index
        temperature_series = reporting_data.df['temperature']
        model_prediction = self.model.predict(
            prediction_index, temperature_series
        )

        df_res = pd.concat([reporting_data.df, model_prediction.result], axis=1)
        df_res = df_res[["temperature", "observed", "predicted_usage"]]
        df_res = df_res.rename(columns={"predicted_usage": "predicted"})
        df_res["predicted_uncertainty"] = np.nan

        # if observed isn't all nan, calculate uncertainty
        if not df_res["observed"].isna().all():
            for month_n, unc_vars in self._autocorr_unc_vars.items():
                if month_n == "all":
                    idx = df_res.index
                else:
                    idx = df_res.index[df_res.index.month == month_n]                   

                mean_baseline_usage = unc_vars["mean_baseline_usage"]
                n = unc_vars["n"]
                n_prime = unc_vars["n_prime"]
                mse = unc_vars["MSE"]

                reporting_usage = np.sum(df_res.loc[idx]["observed"])
                m = len(idx)
                t = t_stat(self.alpha, m, tail=2)

                # ASHRAE 14
                total_unc = 1.26*t*reporting_usage/(m*mean_baseline_usage)*np.sqrt(
                    mse*n/n_prime*(1 + 2/n_prime)*m
                )

                avg_unc = np.sqrt(total_unc**2/m)
                df_res.loc[idx, "predicted_uncertainty"] = avg_unc

        return df_res

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