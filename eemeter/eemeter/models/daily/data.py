import eemeter.common.const as _const
from eemeter.common.abstract_data_processor import AbstractDataProcessor
from eemeter.eemeter.common.data_processor_utilities import as_freq, caltrack_sufficiency_criteria_baseline, clean_caltrack_billing_daily_data, compute_minimum_granularity
from eemeter import compute_temperature_features
from eemeter.eemeter.models.daily.utilities.config import DailySettings
from eemeter.eemeter.warnings import EEMeterWarning

import numpy as np
import pandas as pd

from typing import Optional


class DailyBaselineData(AbstractDataProcessor):
    """Data processor for daily data.

    2.2.1.4. Values of 0 are considered missing for electricity data, but not gas data.

    """
    def __init__(self, data : pd.DataFrame, is_electricity_data : bool, settings : Optional[DailySettings] = None):
        # Because the init method has some specific initializations for Daily / Billing
        """Initialize the data processor.

        Parameters
        ----------
        settings : DailySettings
            Settings for the data processor.
        """
        if settings is None:
            self._settings = DailySettings()
        else:
            self._settings = settings

        self._baseline_meter_df = None
        self.warnings = None
        self.disqualification = None
        self.is_electricity_data = is_electricity_data

        self.set_data(data = data, is_electricity_data = is_electricity_data)


    def _check_data_sufficiency(self, df : pd.DataFrame):
        """
            https://docs.caltrack.org/en/latest/methods.html#section-2-data-management
            Check under section 2.2 : Data constraints
        """
        self.warnings = []
        
        """
            2.2.2.1. If summing to daily usage from higher frequency interval data, no more than 50% of high-frequency values should be missing. 
            Missing values should be filled in with average of non-missing values (e.g., for hourly data, 24 * average hourly usage).
        """
        meter_series = df['observed'].dropna()
        min_granularity = compute_minimum_granularity(meter_series.index)
        if min_granularity.startswith('billing'):
            # TODO : make this a warning instead of an exception
            raise ValueError("Billing data is not allowed in the daily model")
        meter_value_df = clean_caltrack_billing_daily_data(meter_series, min_granularity, self.warnings)
        meter_value_df = meter_value_df.rename(columns={'value': 'observed'})

        temp_series = df['temperature']
        temp_series.index.freq = temp_series.index.inferred_freq
        if temp_series.index.freq != 'H':
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency",
                    description=("Cannot confirm that pre-aggregated temperature data had sufficient hours kept"),
                    data={},
                )
            )
            # TODO consider disallowing this until a later patch
            temperature_features = temp_series.to_frame()
            temperature_features['temperature_null'] = temp_series.isnull().astype(int)
            temperature_features['temperature_not_null'] = temp_series.notnull().astype(int)
            temperature_features['n_days_kept'] = 0  # unused
            temperature_features['n_days_dropped'] = 0  # unused
        else:
            temperature_features = compute_temperature_features(
                meter_value_df.index,
                temp_series,
                data_quality=True,
            )
        criteria_df = meter_value_df.merge(temperature_features, left_index=True, right_index=True, how='outer')
        criteria_df = criteria_df.rename({'temperature_mean': 'temperature'}, axis=1)
        df, self.disqualification, warnings = caltrack_sufficiency_criteria_baseline(criteria_df)
        self.warnings += warnings

        # Add Season and Weekday_weekend 
        df['season'] = df.index.month_name().map(_const.default_season_def)
        df['weekday_weekend'] = df.index.day_name().map(_const.default_weekday_weekend_def)

        # drop data quality columns
        df = df.drop(columns=['temperature_null', 'temperature_not_null', 'n_days_kept', 'n_days_dropped'])

        return df


    def set_data(self, data : pd.DataFrame, is_electricity_data : bool):
        """Process data input for the Daily Model Baseline Class
        Datetime has to be either index or a separate column in the dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            Required columns - datetime, meter value, temperature mean

        Returns
        -------
        processed_data : pd.DataFrame
            Dataframe appended with the correct season and day of week.
        """

        expected_columns = ["observed", "temperature"]
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

        # Convert electricity data having 0 meter values to NaNs
        if is_electricity_data:
            df.loc[df['observed'] == 0, 'observed'] = np.nan


        # Data Sufficiency Check
        df = self._check_data_sufficiency(df)

        # TODO : how to handle the warnings? Should we throw an exception or just print the warnings?
        if self.disqualification or self.warnings:
            for warning in self.disqualification + self.warnings:
                print(warning.json())

        self._baseline_meter_df = df


class DailyReportingData(AbstractDataProcessor):
    """
        Refer to Section 3.5 in https://docs.caltrack.org/en/latest/methods.html#section-2-data-management

        The Set data will be very similar (might be the exact same) as the Baseline version of this class. The only difference will be
        the data_sufficiency check. Although that will also be reused.
    """
    def __init__(self, data : pd.DataFrame, settings : Optional[DailySettings] = None):
        """Initialize the data processor.

        Parameters
        ----------
        settings : DailySettings
            Settings for the data processor.
        """
        if settings is None:
            self._settings = DailySettings()
        else:
            self._settings = settings

        self._reporting_meter_df = None
        self.warnings = None
        self.disqualification = None

        # TODO : do we need to set electric data for reporting?
        self.set_data(data = data, is_electricity_data = False)


    def _check_data_sufficiency(self, df : pd.DataFrame):
        """Check data sufficiency for the given meter.

        3.5.2.1. If a day is missing a consumption value, the corresponding counterfactual value for that day should be masked.
        3.5.3.1. Counterfactual usage is not calculated when daily temperature data is missing, pending further methodological discussion.

        3.5.2.3. Values of 0 are considered missing for electricity data, but not gas data.
        Take input parameter is_electricity_data to determine whether to check for 0 values.

        Parameters
        ----------
        data : pd.DataFrame
            Data to check.

        Returns
        -------
        is_sufficient : bool
            Whether the data is sufficient.
        """
        
        df['temperature_null'] = df.temperature.isnull().astype(int)
        df['temperature_not_null'] = df.temperature.notnull().astype(int)

        df, self.disqualification, self.warnings = caltrack_sufficiency_criteria_baseline(data = df, is_reporting_data = True)

        df = as_freq(df['temperature'], 'D', series_type = 'instantaneous').to_frame(name='temperature')

        # TODO : interpolate if necessary

        return df

    def _interpolate_data(self, data):
        # TODO : Is this even required? Or just throw a warning if we don't have the reporting data?
        pass


    def set_data(self, data : pd.DataFrame, is_electricity_data : bool):
        """Process reporting data. This will be very similar to the Baseline version of this method.

        Parameters
        ----------
        data : pd.DataFrame
            Data to process.

        Returns
        -------
        processed_data : pd.DataFrame
            Processed data.
        """

        if 'temperature' not in data.columns:
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