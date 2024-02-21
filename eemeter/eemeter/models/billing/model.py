from typing import Union

import numpy as np
import pandas as pd

from eemeter.eemeter.models.daily.model import DailyModel, CVRMSE_THRESHOLD
from eemeter.eemeter.models.billing.data import BillingBaselineData, BillingReportingData
from eemeter.eemeter.models.billing.plot import plot
from eemeter.eemeter.exceptions import DataSufficiencyError
from eemeter.eemeter.warnings import EEMeterWarning


class BillingModel(DailyModel): 
    """wrapper for DailyModel using billing presets"""

    def __init__(self, settings=None):
        super().__init__(model="2.0", settings=settings)

    def fit(self, baseline_data: BillingBaselineData, ignore_disqualification=False):
        #TODO there's a fair bit of duplicated code between this and daily fit(), refactor
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
        self._fit(baseline_data.df)
        if self.error["CVRMSE"] > CVRMSE_THRESHOLD:
            cvrmse_warning = EEMeterWarning(
                qualified_name="eemeter.model_fit_metrics.cvrmse",
                description=(f"Fit model has CVRMSE > {CVRMSE_THRESHOLD}"),
                data={"CVRMSE": self.error["CVRMSE"]}
            )
            cvrmse_warning.warn()
            self.disqualification.append(cvrmse_warning)
        return self

    def predict(self, reporting_data: Union[BillingBaselineData, BillingReportingData], aggregation=None):
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")

        if not isinstance(reporting_data, (BillingBaselineData, BillingReportingData)):
            raise TypeError("reporting_data must be a BillingBaselineData or BillingReportingData object")
        
        df_res = self._predict(reporting_data.df)

        if aggregation is None:
            agg = None
        elif aggregation.lower() == "none":
            agg = None
        elif aggregation == "monthly":
            agg = "MS"
        elif aggregation == "bimonthly":
            agg = "2MS"
        else:
            raise ValueError("aggregation must be one of [None, 'monthly', 'bimonthly']")

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

            df_res = pd.concat([season, temperature, observed, predicted, predicted_unc,
                                heating_load, cooling_load, model_split, model_type], axis=1)

        return df_res
    
    def plot(
        self,
        df_eval,
        ax=None,
        title=None,
        figsize=None,
        temp_range=None,
        aggregation=None,
    ):
        """Plot a model fit.

        Parameters
        ----------
        ax : :any:`matplotlib.axes.Axes`, optional
            Existing axes to plot on.
        title : :any:`str`, optional
            Chart title.
        figsize : :any:`tuple`, optional
            (width, height) of chart.
        with_candidates : :any:`bool`
            If True, also plot candidate models.
        temp_range : :any:`tuple`, optionl
            Temperature range to plot

        Returns
        -------
        ax : :any:`matplotlib.axes.Axes`
            Matplotlib axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting.")

        #TODO: pass more kwargs to plotting function

        plot(self, self.predict(df_eval, aggregation=aggregation))

    def to_dict(self):
        model_dict = super().to_dict()
        model_dict['settings']['developer_mode'] = True
        return model_dict
