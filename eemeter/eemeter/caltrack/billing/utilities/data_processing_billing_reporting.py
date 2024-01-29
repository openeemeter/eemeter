from eemeter.common.abstract_data_processor import AbstractDataProcessor
from eemeter.common.data_settings import MonthlySettings
import pandas as pd

class BillingDataProcessorReporting(AbstractDataProcessor):
    def __init__(self, settings : MonthlySettings | None):
        """Initialize the data processor.
        
        Parameters
        ----------
        settings : DailySettings
            Settings for the data processor.
        """
        if settings is None:
            self._settings = MonthlySettings()
        else:
            self._settings = settings

        self._baseline_meter_df = None
        self._reporting_meter_df = None


    def _check_data_sufficiency(self, data : pd.DataFrame):

        # TODO : reuse the eemeter.clean_caltrack_billing_daily_data() method instead of rewriting

        pass

    def set_data(self, data : pd.DataFrame, is_electricity_data : bool):
        """Process data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to process.

        Returns
        -------
        processed_data : pd.DataFrame
            Processed data.
        """

        # TODO : try and abstract out the baseline version instead of copying it here
        pass

    def extend(self, df):
        """Extend data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to extend.

        Returns
        -------
        extended_data : pd.DataFrame
        """
        pass