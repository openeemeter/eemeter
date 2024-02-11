import eemeter.common.const as _const
from eemeter.common.abstract_data_processor import AbstractDataProcessor
from eemeter.common.data_settings import MonthlySettings
from eemeter.eemeter.common.data_processor_utilities import as_freq, caltrack_sufficiency_criteria_baseline, clean_caltrack_billing_daily_data, compute_minimum_granularity


import numpy as np
import pandas as pd


class DataBillingBaseline(AbstractDataProcessor):
    """Baseline data processor for billing data.

    2.2.3.4. Off-cycle reads (spanning less than 25 days) should be dropped from analysis. 
    These readings typically occur due to meter reading problems or changes in occupancy.

    2.2.3.5. For pseudo-monthly billing cycles, periods spanning more than 35 days should be dropped from analysis. 
    For bi-monthly billing cycles, periods spanning more than 70 days should be dropped from the analysis.
    """

    def __init__(self, data : pd.DataFrame, is_electricity_data, settings : MonthlySettings | None = None):
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
        self.warnings = None
        self.disqualification = None
        self.is_electricity_data = is_electricity_data

        self.set_data(data = data, is_electricity_data = is_electricity_data)


    def _check_data_sufficiency(self, df : pd.DataFrame):
        # Add Season and Weekday_weekend columns
        df['season'] = df.index.month_name().map(_const.default_season_def)
        df['weekday_weekend'] = df.index.day_name().map(_const.default_weekday_weekend_def)

        df['temperature_null'] = df.temperature_mean.isnull().astype(int)
        df['temperature_not_null'] = df.temperature_mean.notnull().astype(int)

        df, self.disqualification, self.warnings = caltrack_sufficiency_criteria_baseline(data = df)

        # TODO : Assume if the billing cycle is mixed between monthly and bimonthly, then the minimum granularity is bimonthly
        # Test for more than 50% of high frequency data being missing
        """
            2.2.2.1. If summing to daily usage from higher frequency interval data, no more than 50% of high-frequency values should be missing. 
            Missing values should be filled in with average of non-missing values (e.g., for hourly data, 24 * average hourly usage).
        """
        min_granularity = compute_minimum_granularity(df.index)

        # Ensure higher frequency data is aggregated to the monthly model
        if not min_granularity.startswith('billing'):
            min_granularity = 'billing_monthly'

        meter_value_df = clean_caltrack_billing_daily_data(df['meter_value'], min_granularity, self.warnings)
        temperature_df = as_freq(df['temperature_mean'], 'M', series_type = 'instantaneous').to_frame(name='temperature_mean')

        # Perform a join
        meter_value_df = meter_value_df.merge(temperature_df, left_index=True, right_index=True, how='outer')

        df = meter_value_df
        return df

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


        # Check that the datetime index is timezone aware timestamp
        if not isinstance(data.index, pd.DatetimeIndex) and 'datetime' not in data.columns:
            raise ValueError("Index is not datetime and datetime not provided")

        elif 'datetime' in data.columns:
            if data['datetime'].dt.tz is None:
                raise ValueError("Datatime is missing timezone information")
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)

        elif data.index.tz is None:
            raise ValueError("Datatime is missing timezone information")


        # Copy the input dataframe so that the original is not modified
        df = data.copy()

        if is_electricity_data:
            df.loc[df['meter_value'] == 0, 'meter_value'] = np.nan

        # Data Sufficiency Check
        df = self._check_data_sufficiency(df)
        # TODO : how to handle the warnings? Should we throw an exception or just print the warnings?
        if self.disqualification or self.warnings:
            for warning in self.disqualification + self.warnings:
                print(warning.json())


        # TODO : Do we need to downsample the daily data for monthly models?
        self._baseline_meter_df = df


class DataBillingReporting(AbstractDataProcessor):
    def __init__(self, data : pd.DataFrame, settings : MonthlySettings | None = None):
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

        self._reporting_meter_df = None
        self.warnings = None
        self.disqualification = None

        # TODO : do we need to set electric data for reporting?
        self.set_data(data = data, is_electricity_data = False)


    def _check_data_sufficiency(self, df : pd.DataFrame):

        df['temperature_null'] = df.temperature_mean.isnull().astype(int)
        df['temperature_not_null'] = df.temperature_mean.notnull().astype(int)

        df, self.disqualification, self.warnings = caltrack_sufficiency_criteria_baseline(data = df, is_reporting_data = True)

        df = as_freq(df['temperature_mean'], 'M', series_type = 'instantaneous').to_frame(name='temperature_mean')

        return df

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

        if 'temperature_mean' not in data.columns:
            raise ValueError("Temperature data is missing")

        # Check that the datetime index is timezone aware timestamp
        if not isinstance(data.index, pd.DatetimeIndex) and 'datetime' not in data.columns:
            raise ValueError("Index is not datetime and datetime not provided")

        elif 'datetime' in data.columns:
            if data['datetime'].dt.tz is None:
                raise ValueError("Datatime is missing timezone information")
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)

        elif data.index.tz is None:
            raise ValueError("Datatime is missing timezone information")


        # Copy the input dataframe so that the original is not modified
        df = data.copy()

        df = self._check_data_sufficiency(df)

        if self.disqualification or self.warnings:
            for warning in self.disqualification + self.warnings:
                print(warning.json())

        self._reporting_meter_df = df