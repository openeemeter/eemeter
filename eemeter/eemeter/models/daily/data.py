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
import eemeter.common.const as _const
from eemeter.eemeter.common.data_processor_utilities import as_freq, caltrack_sufficiency_criteria_baseline, clean_caltrack_billing_daily_data, compute_minimum_granularity
from eemeter.eemeter.common.features import compute_temperature_features
from eemeter.eemeter.common.warnings import EEMeterWarning

import numpy as np
import pandas as pd
from datetime import datetime

from typing import Optional, Union


class _DailyData:
    """Private base class for daily baseline and reporting data.
    
    Will raise exception during data sufficiency check if instantiated
    """

    def __init__(self, df: pd.DataFrame, is_electricity_data: bool):
        self._df = None
        self.warnings = []
        self.disqualification = []
        self.is_electricity_data = is_electricity_data
        self.tz = None

        #TODO re-examine dq/warning pattern. keep consistent between
        # either implicitly setting as side effects, or returning and assigning outside
        self._df, temp_coverage = self._set_data(df)

        sufficiency_df = self._df.merge(temp_coverage, left_index=True, right_index=True, how='left')
        disqualification, warnings = self._check_data_sufficiency(sufficiency_df)

        self.disqualification += disqualification
        self.warnings += warnings
        self.log_warnings()

    @property
    def df(self):
        """
        Get the corrected input data stored in the class. The actual dataframe is immutable, this returns a copy.

        Returns:
            pandas.DataFrame or None: A copy of the DataFrame if it exists, otherwise None.
        """
        if self._df is None:
            return None
        else:
            return self._df.copy()
    
    @classmethod
    def from_series(cls, meter_data: Union[pd.Series, pd.DataFrame], temperature_data: Union[pd.Series, pd.DataFrame], is_electricity_data):
        """
        Create an instance of the Data class from meter data and temperature data.

        Parameters:
        - meter_data (pd.Series or pd.DataFrame): The meter data.
        - temperature_data (pd.Series or pd.DataFrame): The temperature data.
        - is_electricity_data: A flag indicating whether the data represents electricity data. This is required as electricity data with 0 values are converted to NaNs.

        Returns:
        - Data: An instance of the Data class with the dataframe populated with the corrected data, alongwith warnings and disqualifications based on the input.
        """
        if isinstance(meter_data, pd.Series):
            meter_data = meter_data.to_frame()
        if isinstance(temperature_data, pd.Series):
            temperature_data = temperature_data.to_frame()
        meter_data = meter_data.rename(columns={meter_data.columns[0]: 'observed'})
        temperature_data = temperature_data.rename(columns={temperature_data.columns[0]: 'temperature'})
        temperature_data.index = temperature_data.index.tz_convert(meter_data.index.tzinfo)
        df = pd.concat([meter_data, temperature_data], axis=1)
        return cls(df, is_electricity_data)

    def log_warnings(self):
        """
        Logs the warnings and disqualifications associated with the data.

        """
        for warning in self.warnings + self.disqualification:
            warning.warn()

    def _compute_meter_value_df(self, df : pd.DataFrame):
        """
        Computes the meter value DataFrame by cleaning and processing the observed meter data.
        1. The minimum granularity is computed from the non null rows.
        2. The meter data is cleaned and downsampled/upsampled into the correct frequency using clean_caltrack_billing_daily_data()
        3. Add missing days as NaN by merging with a full year daily index.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the observed meter data.

        Returns:
            pd.DataFrame: The cleaned and processed meter value DataFrame.
        """
        # Dropping the NaNs is beneficial when the meter data is spread over hourly temperature data, causing lots of NaNs
        # But causes problems in detection of frequency when there are genuine missing values. The missing days are accounted for in the sufficiency_criteria_baseline method
        # whereas they should actually be kept.
        meter_series = df['observed'].dropna()
        if meter_series.empty:
            return df['observed'].resample('D').first().to_frame()
        min_granularity = compute_minimum_granularity(meter_series.index, 'daily')
        if min_granularity.startswith('billing'):
            # TODO : make this a warning instead of an exception
            raise ValueError("Billing data is not allowed in the daily model")
        meter_value_df = clean_caltrack_billing_daily_data(meter_series, min_granularity, self.warnings)
        # if np.isnan(meter_value_df.iloc[-1]['value']):
        #     #TODO test behavior here. we might be able to get away with a dropna() if we alter the n_days check, but this is less aggressive
        #     #TODO need to refactor the clean caltrack data method and remove the nan logic, this breaks when original df has nan in last row
        #     meter_value_df = meter_value_df[:-1]

        meter_value_df = meter_value_df.rename(columns={'value': 'observed'})
        meter_value_df.index = meter_value_df.index.normalize() # in case of daily data where hour is not midnight

        # To account for the above issue, we create an index with all the days and then merge the meter_value_df with it
        # This will ensure that the missing days are kept in the dataframe
        # Create an index with all the days from the start and end date of 'meter_value_df'
        all_days_index = pd.date_range(start=meter_value_df.index.min(), end=meter_value_df.index.max(), freq='D', tz=df.index.tz)
        all_days_df = pd.DataFrame(index=all_days_index)
        meter_value_df = meter_value_df.merge(all_days_df, left_index=True, right_index=True, how='outer')

        return meter_value_df

    def _compute_temperature_features(self, df: pd.DataFrame, meter_index: pd.DatetimeIndex):
        """
        Compute temperature features for the given DataFrame and meter index.
        1. The frequency of the temperature data is inferred and set to hourly if not already. If frequency is not inferred or its lower than hourly, a warning is added.
        2. The temperature data is downsampled/upsampled into the daily frequency using as_freq()
        3. High frequency temperature data is checked for missing values and a warning is added if more than 50% of the data is missing, and those rows are set to NaN.
        4. If frequency was already hourly, compute_temperature_features() is used to recompute the temperature to match with the meter index.

        Args:
            df (pd.DataFrame): The DataFrame containing temperature data.
            meter_index (pd.DatetimeIndex): The meter index.

        Returns:
            pd.Series: The computed temperature values.
            pd.DataFrame: The computed temperature features.
        """
        temp_series = df['temperature']
        temp_series.index.freq = temp_series.index.inferred_freq
        if temp_series.index.freq != 'H':
            if temp_series.index.freq is None or temp_series.index.freq > pd.Timedelta(hours=1):
                # Add warning for frequencies longer than 1 hour
                self.warnings.append(
                    EEMeterWarning(
                        qualified_name="eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency",
                        description=("Cannot confirm that pre-aggregated temperature data had sufficient hours kept"),
                        data={},
                    )
                )
            if temp_series.index.freq != 'D':
                # Downsample / Upsample the temperature data to daily
                temperature_features = as_freq(temp_series, 'D', series_type = 'instantaneous', include_coverage = True)
                # If high frequency data check for 50% data coverage in rollup
                if len(temperature_features[temperature_features.coverage <= 0.5]) > 0:
                    self.warnings.append(
                        EEMeterWarning(
                            qualified_name="eemeter.caltrack_sufficiency_criteria.missing_high_frequency_temperature_data",
                            description=("More than 50% of the high frequency Temperature data is missing."),
                            data = {
                                'high_frequency_data_missing_count' : len(temperature_features[temperature_features.coverage <= 0.5].index.to_list())
                            }
                        )
                    )

                # Set missing high frequency data to NaN
                temperature_features.value[temperature_features.coverage > 0.5] = (
                    temperature_features[temperature_features.coverage > 0.5].value / temperature_features[temperature_features.coverage > 0.5].coverage
                )

                temperature_features = temperature_features[temperature_features.coverage > 0.5].reindex(temperature_features.index)[["value"]].rename(columns={'value' : 'temperature_mean'})
                
                if 'coverage' in temperature_features.columns:
                    temperature_features = temperature_features.drop(columns=['coverage'])
            else:
                temperature_features = temp_series.to_frame(name='temperature_mean')

            temperature_features['temperature_null'] = temp_series.isnull().astype(int)
            temperature_features['temperature_not_null'] = temp_series.notnull().astype(int)
            temperature_features['n_days_kept'] = 0  # unused
            temperature_features['n_days_dropped'] = 0  # unused
        else:
            #TODO hacky method of avoiding the last index nan convention
            buffer_idx = pd.to_datetime('2090-01-01 00:00:00+00:00').tz_convert(meter_index.tz)

            temperature_features = compute_temperature_features(
                meter_index.union([buffer_idx]),
                temp_series,
                data_quality=True,
            )
            temperature_features = temperature_features[:-1]

            # Only check for high frequency temperature data if it exists
            if (temperature_features.temperature_not_null + temperature_features.temperature_null).median() > 1:
                invalid_temperature_rows = (
                    temperature_features.temperature_not_null
                    / (temperature_features.temperature_not_null + temperature_features.temperature_null)
                ) <= 0.5

                # Set high frequency temperature data with more than 50% data missing as NaN
                if invalid_temperature_rows.any():
                    self.warnings.append(
                        EEMeterWarning(
                            qualified_name="eemeter.caltrack_sufficiency_criteria.missing_high_frequency_temperature_data",
                            description=("More than 50% of the high frequency temperature data is missing."),
                            data = (
                                invalid_temperature_rows.index.to_list()
                            )
                        )
                    )
                    temperature_features.loc[invalid_temperature_rows, 'temperature_mean'] = np.nan

        temp = temperature_features['temperature_mean'].rename('temperature')
        features = temperature_features.drop(columns=['temperature_mean'])
        return temp, features
    
    def _merge_meter_temp(self, meter, temp):
        """
        Merge the meter and temperature dataframes and reorder the columns to have the order -
            [season, weekday_weekend, temperature, observed (if present)]

        Args:
            meter (pd.DataFrame): The meter dataframe.
            temp (pd.DataFrame): The temperature dataframe.

        Returns:
            pd.DataFrame: The merged and transformed dataframe.
        """
        df = meter.merge(temp, left_index=True, right_index=True, how='left').tz_convert(meter.index.tz)
        if df['observed'].dropna().empty:
            df = df.drop(columns=['observed'])

        # Add Season and Weekday_weekend 
        df['season'] = df.index.month_name().map(_const.default_season_def)
        df['weekday_weekend'] = df.index.day_name().map(_const.default_weekday_weekend_def)

        # Reorder the columns Create a list of columns
        columns = ['season', 'weekday_weekend', 'temperature']
        if 'observed' in df.columns:
            columns.append('observed')
        df = df[columns]

        return df
        
    def _check_data_sufficiency(self, sufficiency_df):
        raise NotImplementedError("Can't instantiate class _DailyData, use DailyBaselineData or DailyReportingData.")

    def _set_data(self, data: pd.DataFrame):
        """Process data input for the Daily Model Baseline Class
        Datetime has to be either index or a separate column in the dataframe.
        Electricity data with 0 meter values are converted to NaNs.

        Parameters
        ----------
        data : pd.DataFrame
            Required columns - datetime, observed, temperature

        Returns
        -------
        processed_data : pd.DataFrame
            Dataframe appended with the correct season and day of week.
        """

        # Copy the input dataframe so that the original is not modified
        df = data.copy()

        expected_columns = ["observed", "temperature"] #TODO will change when extending to report
        #TODO maybe check datatypes
        if not set(expected_columns).issubset(set(df.columns)):
            # show the columns that are missing

            raise ValueError("Data is missing required columns: {}".format(
                set(expected_columns) - set(df.columns)))

        # Check that the datetime index is timezone aware timestamp
        if not isinstance(df.index, pd.DatetimeIndex) and 'datetime' not in df.columns:
            raise ValueError("Index is not datetime and datetime not provided")

        elif 'datetime' in df.columns:
            if df['datetime'].dt.tz is None:
                raise ValueError("Datatime is missing timezone information")
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

        elif df.index.tz is None:
            raise ValueError("Datatime is missing timezone information")
        elif str(df.index.tz) == 'UTC':
            self.warnings.append(EEMeterWarning(
                qualified_name="eemeter.data_quality.utc_index",
                description=("Datetime index is in UTC. Use tz_localize() with the local timezone to ensure correct aggregations"),
                data={},
            ))
        self.tz = df.index.tz

        # Convert electricity data having 0 meter values to NaNs
        if self.is_electricity_data:
            df.loc[df['observed'] == 0, 'observed'] = np.nan

        meter = self._compute_meter_value_df(df)
        temp, temp_coverage = self._compute_temperature_features(df, meter.index)
        final_df = self._merge_meter_temp(meter, temp)
        return final_df, temp_coverage


class DailyBaselineData(_DailyData):
    """Dataclass for daily baseline data"""
    def _check_data_sufficiency(self, sufficiency_df):
        # 90% coverage per period only required for billing models
        _, disqualification, warnings = caltrack_sufficiency_criteria_baseline(sufficiency_df, min_fraction_hourly_temperature_coverage_per_period=0.5, is_reporting_data=False, is_electricity_data=self.is_electricity_data)
        return disqualification, warnings


class DailyReportingData(_DailyData):
    """Dataclass for daily reporting data
    
    Uses looser criteria to determine sufficiency
    Adds ability to omit meter_data for predictions
    """
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool):
        if 'observed' not in df.columns:
            df = df.copy()
            df['observed'] = np.nan
        super().__init__(df, is_electricity_data)        
        
    @classmethod
    def from_series(cls, meter_data: Optional[Union[pd.Series, pd.DataFrame]], temperature_data: Union[pd.Series, pd.DataFrame], is_electricity_data: Optional[bool]=None, tzinfo=None):
        if tzinfo and meter_data is not None:
            raise ValueError('When passing meter data to DailyReportingData, convert its DatetimeIndex to local timezone first; `tzinfo` param should only be used in the absence of reporting meter data.')
        if is_electricity_data is None and meter_data is not None:
            raise ValueError('Must specify is_electricity_data when passing meter data.')
        if meter_data is None:
            meter_data = pd.DataFrame({'observed': np.nan}, index=temperature_data.index)
            if tzinfo:
                meter_data = meter_data.tz_convert(tzinfo)
        return super().from_series(meter_data, temperature_data, is_electricity_data)

    def _check_data_sufficiency(self, sufficiency_df):
        # 90% coverage per period only required for billing models
        _, disqualification, warnings = caltrack_sufficiency_criteria_baseline(sufficiency_df, min_fraction_hourly_temperature_coverage_per_period=0.5, is_reporting_data=True, is_electricity_data=self.is_electricity_data)
        return disqualification, warnings           