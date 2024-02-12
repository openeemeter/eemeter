from typing import Union

from eemeter.eemeter.models.daily.model import DailyModel
from eemeter.eemeter.models.billing.data import BillingBaselineData, BillingReportingData
from eemeter.eemeter.exceptions import DataSufficiencyError


class BillingModel(DailyModel): 
    """wrapper for DailyModel using billing presets"""

    def __init__(self, settings=None):
        super().__init__(model="2.0", settings=settings)

    def fit(self, baseline_data: BillingBaselineData, ignore_disqualification=False):
        if not isinstance(baseline_data, BillingBaselineData):
            raise TypeError("baseline_data must be a BillingBaselineData object")
        if baseline_data.disqualification and not ignore_disqualification:
            for warning in baseline_data.disqualification + baseline_data.warnings:
                print(warning.json())
            raise DataSufficiencyError("Can't fit model on disqualified baseline data")
        self.warnings = baseline_data.warnings
        self.disqualification = baseline_data.disqualification
        for warning in baseline_data.warnings + baseline_data.disqualification:
            print(warning.json())
        meter_data = baseline_data._baseline_meter_df #TODO make df public attr
        return self._fit(meter_data)

    def predict(self, reporting_data: Union[BillingBaselineData, BillingReportingData]):
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")

        if not isinstance(reporting_data, (BillingBaselineData, BillingReportingData)):
            raise TypeError("reporting_data must be a BillingBaselineData or BillingReportingData object")

        if isinstance(reporting_data, BillingBaselineData):
            df_eval = reporting_data._baseline_meter_df
        if isinstance(reporting_data, BillingReportingData):
            df_eval = reporting_data._reporting_meter_df 
        return self._predict(df_eval)