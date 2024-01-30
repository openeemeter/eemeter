from eemeter.common.abstract_data_processor import AbstractDataProcessor
import eemeter.common.const as _const
from eemeter.common.data_settings import MonthlySettings
import numpy as np
import pandas as pd

class DataProcessorBillingBaseline(AbstractDataProcessor):
    """Baseline data processor for billing data.

    2.2.3.4. Off-cycle reads (spanning less than 25 days) should be dropped from analysis. 
    These readings typically occur due to meter reading problems or changes in occupancy.

    2.2.3.5. For pseudo-monthly billing cycles, periods spanning more than 35 days should be dropped from analysis. 
    For bi-monthly billing cycles, periods spanning more than 70 days should be dropped from the analysis.
    """
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
        # s = self._settings
        # descriptions = []
        # details = []

        # def _update_from_n_days_kept():
        #     if self.data_sufficiency is None:
        #         descriptions.append(
        #             "unable to verify n_days_kept requirement due to missing data_sufficiency field"
        #         )
        #         details.append(
        #             "unable to verify n_days_kept requirement due to missing data_sufficiency field"
        #         )
        #         return

        #     if self.data_sufficiency.n_days_kept is None:
        #         descriptions.append(
        #             "n_days_kept is None. likely means a daily model was unable to be created"
        #         )
        #         details.append(
        #             "n_days_kept is None. likely means a daily model was unable to be created"
        #         )
        #         return

        #     if self.data_sufficiency.n_days_kept < s.n_days_kept_min:
        #         descriptions.append(
        #             f"n_days_kept requirment of {s.n_days_kept_min} not met"
        #         )
        #         details.append(
        #             f"n_days_kept min: {s.n_days_kept_min}; received {self.data_sufficiency.n_days_kept}"
        #         )
        #         return

        # TODO : reuse the code in /app/eemeter/eemeter/features.py compute_time_features() method
        pass

    def set_data(self, data : pd.DataFrame, is_electricity_data : bool):
        """Process data for the monthly / billing case

        Parameters
        ----------
        data : pd.DataFrame
            Data to process.

        Returns
        -------
        processed_data : pd.DataFrame
            Processed data.
        """
        expected_columns = ["meter_value", "temperature_mean"]
        if not set(expected_columns).issubset(set(data.columns)):
            # show the columns that are missing

            raise ValueError("Data is missing required columns: {}".format(
                set(expected_columns) - set(data.columns)))

        # TODO : Handle the case if the datetime is not the index but provided in a separate column

        # Check that the datetime index is timezone aware timestamp
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Index is not datetime")
        elif data.index.tz is None:
            raise ValueError("Datatime is missing timezone information")
        

        # Copy the input dataframe so that the original is not modified
        df = data.copy()

        # TODO : Check missing value in datetime and add a warning/ exception if missing

        if not is_electricity_data:
            df.loc[df['meter_value'] == 0, 'meter_value'] = np.nan

        # Data Sufficiency Check
        self._check_data_sufficiency(df)
        self._baseline_meter_df = df
        
        if self._sufficiency_warnings is not None:
            # TODO : how to handle the warnings?
            print(self._sufficiency_warnings)


        # TODO : Rollup the data, resample?

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