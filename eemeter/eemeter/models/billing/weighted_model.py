#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A module housing Billing Model classes and functions.

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
from __future__ import annotations

import numpy as np
import pandas as pd

from eemeter.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
from eemeter.eemeter.common.warnings import EEMeterWarning
from eemeter.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)
from eemeter.eemeter.models.billing.settings import BillingSettings
from eemeter.eemeter.models.daily.model import DailyModel


class BillingWeightedModel(DailyModel):
    """A class to fit a model to the input meter data.

    BillingModel is a wrapper for the DailyModel class using billing presets.

    Attributes:
        settings (dict): A dictionary of settings.
        seasonal_options (list): A list of seasonal options (su: Summer, sh: Shoulder, wi: Winter).
            Elements in the list are seasons separated by '_' that represent a model split.
            For example, a list of ['su_sh', 'wi'] represents two splits: summer/shoulder and winter.
        day_options (list): A list of day options.
        combo_dictionary (dict): A dictionary of combinations.
        df_meter (pandas.DataFrame): A dataframe of meter data.
        error (dict): A dictionary of error metrics.
        combinations (list): A list of combinations.
        components (list): A list of components.
        fit_components (list): A list of fit components.
        wRMSE_base (float): The mean bias error for no splits.
        best_combination (list): The best combination of splits.
        model (sklearn.pipeline.Pipeline): The final fitted model.
        id (str): The index of the meter data.
    """

    # TODO: lot of duplicated code between this and daily model, refactor later
    def __init__(
        self,
        settings: dict | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            settings: DailySettings to be changed.
            verbose: Whether to print verbose output.
        """

        # Initialize settings
        # Note: Model designates the base settings, it can be 'current' or 'legacy'
        #       Settings is to be a dictionary of settings to be changed

        if settings is None:
            settings = {}

        self.settings = BillingSettings(**settings)

        # Initialize seasons and weekday/weekend
        self.seasonal_options = [
            ["su_sh_wi"],
            ["su", "sh_wi"],
            ["su_sh", "wi"],
            ["su_wi", "sh"],
            ["su", "sh", "wi"],
        ]
        self.day_options = [["wd", "we"]]

        # make dictionary is_weekday from settings
        day_dict = self.settings.weekday_weekend._num_dict
        n_week = list(range(len(day_dict)))
        self.combo_dictionary = {
            "su": "summer",
            "sh": "shoulder",
            "wi": "winter",
            "fw": [n + 1 for n in n_week],
            "wd": [n + 1 for n in n_week if day_dict[n+1] == "weekday"],
            "we": [n + 1 for n in n_week if day_dict[n+1] == "weekend"],
        }
        self.verbose = verbose

        self.error = {
            "wRMSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "CVRMSE": np.nan,
            "PNRMSE": np.nan,
        }

    def fit(
        self, baseline_data: BillingBaselineData, ignore_disqualification: bool = False
    ) -> BillingWeightedModel:
        """Fit the model using baseline data.

        Args:
            baseline_data: BillingBaselineData object.
            ignore_disqualification: Whether to ignore disqualification errors / warnings.

        Returns:
            The fitted model.

        Raises:
            TypeError: If baseline_data is not a BillingBaselineData object.
            DataSufficiencyError: If the model can't be fit on disqualified baseline data.
        """
        # TODO there's a fair bit of duplicated code between this and daily fit(), refactor
        if not isinstance(baseline_data, BillingBaselineData):
            raise TypeError("baseline_data must be a BillingBaselineData object")
        baseline_data.log_warnings()
        if baseline_data.disqualification and not ignore_disqualification:
            for warning in baseline_data.disqualification + baseline_data.warnings:
                print(warning.json())
            raise DataSufficiencyError("Can't fit model on disqualified baseline data")
        self.baseline_timezone = baseline_data.tz
        self.warnings = baseline_data.warnings
        self.disqualification = baseline_data.disqualification
        self._fit(baseline_data.billing_df)
        if self.error["CVRMSE"] > self.settings.cvrmse_threshold:
            cvrmse_warning = EEMeterWarning(
                qualified_name="eemeter.model_fit_metrics.cvrmse",
                description=(
                    f"Fit model has CVRMSE > {self.settings.cvrmse_threshold}"
                ),
                data={"CVRMSE": self.error["CVRMSE"]},
            )
            cvrmse_warning.warn()
            self.disqualification.append(cvrmse_warning)
        return self

    def _fit(self, meter_data):
        # Initialize dataframe
        self.df_meter, _ = self._initialize_data(meter_data)

        # Begin fitting
        self.combinations = self._combinations()
        self.components = self._components()
        self.fit_components = self._fit_components()

        # calculate mean bias error for no splits
        self.wRMSE_base = self._get_error_metrics("fw-su_sh_wi")[0]

        # find best combination
        self.best_combination = self._best_combination(print_out=False)
        self.model = self._final_fit(self.best_combination)

        self.id = meter_data.index.unique()[0]

        wRMSE, RMSE, MAE, CVRMSE, PNRMSE = self._get_error_metrics(
            self.best_combination
        )
        self.error["wRMSE"] = float(wRMSE)
        self.error["RMSE"] = float(RMSE)
        self.error["MAE"] = float(MAE)
        self.error["CVRMSE"] = float(CVRMSE)
        self.error["PNRMSE"] = float(PNRMSE)

        self.params = self._create_params_from_fit_model()
        self.is_fitted = True
        return self

    def predict(
        self,
        reporting_data: BillingBaselineData | BillingReportingData,
        aggregation: str | None = None,
        ignore_disqualification: bool = False,
    ) -> pd.DataFrame:
        """Predicts the energy consumption using the fitted model.

        Args:
            reporting_data: The data used for prediction.
            aggregation: The aggregation level for the prediction. One of [None, 'none', 'monthly', 'bimonthly'].
            ignore_disqualification: Whether to ignore model disqualification. Defaults to False.

        Returns:
            Dataframe with input data along with predicted energy consumption.

        Raises:
            RuntimeError: If the model is not fitted.
            DisqualifiedModelError: If the model is disqualified and ignore_disqualification is False.
            TypeError: If the reporting data is not of type BillingBaselineData or BillingReportingData.
            ValueError: If the aggregation is not one of [None, 'none', 'monthly', 'bimonthly'].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")

        if self.disqualification and not ignore_disqualification:
            raise DisqualifiedModelError(
                "Attempting to predict using disqualified model without setting ignore_disqualification=True"
            )

        if not isinstance(reporting_data, (BillingBaselineData, BillingReportingData)):
            raise TypeError(
                "reporting_data must be a BillingBaselineData or BillingReportingData object"
            )

        df_res = self._predict(reporting_data.billing_df)

        if aggregation is None:
            agg = None
        elif aggregation.lower() == "none":
            agg = None
        elif aggregation == "monthly":
            agg = "MS"
        elif aggregation == "bimonthly":
            agg = "2MS"
        else:
            raise ValueError(
                "aggregation must be one of [None, 'monthly', 'bimonthly']"
            )

        if agg is not None:
            sum_quad = lambda x: np.sqrt(np.sum(np.square(x)))

            season = df_res["season"].resample(agg).first()
            temperature = df_res["temperature"].resample(agg).mean()
            observed = df_res["observed"].resample(agg).sum()
            predicted = df_res["predicted"].resample(agg).sum()
            predicted_unc = df_res["predicted_unc"].resample(agg).apply(sum_quad)
            heating_load = df_res["heating_load"].resample(agg).sum()
            cooling_load = df_res["cooling_load"].resample(agg).sum()
            model_split = df_res["model_split"].resample(agg).first()
            model_type = df_res["model_type"].resample(agg).first()

            df_res = pd.concat(
                [
                    season,
                    temperature,
                    observed,
                    predicted,
                    predicted_unc,
                    heating_load,
                    cooling_load,
                    model_split,
                    model_type,
                ],
                axis=1,
            )

        return df_res

    def plot(
        self,
        df_eval,
        aggregation: str | None = None,
    ):
        """Plot a model fit with baseline or reporting data. Requires matplotlib to use.

        Args:
            df_eval: The baseline or reporting data object to plot.
            aggregation: The aggregation level for the prediction. One of [None, 'none', 'monthly', 'bimonthly'].
        """
        try:
            from eemeter.eemeter.models.billing.plot import plot
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting.")

        # TODO: pass more kwargs to plotting function

        plot(self, self.predict(df_eval, aggregation=aggregation))

    def to_dict(self) -> dict:
        """Returns a dictionary of model parameters.

        Returns:
            Model parameters.
        """
        model_dict = super().to_dict()
        model_dict["settings"]["developer_mode"] = True

        return model_dict
