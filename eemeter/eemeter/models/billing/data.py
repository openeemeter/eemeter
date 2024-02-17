import eemeter.common.const as _const
from eemeter.common.abstract_data_processor import AbstractDataProcessor
from eemeter.eemeter.common.data_processor_utilities import as_freq, caltrack_sufficiency_criteria_baseline, clean_caltrack_billing_daily_data, compute_minimum_granularity
from eemeter import compute_temperature_features
from eemeter.eemeter.warnings import EEMeterWarning

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from typing import Optional
import warnings


class BillingBaselineData(AbstractDataProcessor):
    """Baseline data processor for billing data.

    2.2.3.4. Off-cycle reads (spanning less than 25 days) should be dropped from analysis. 
    These readings typically occur due to meter reading problems or changes in occupancy.

    2.2.3.5. For pseudo-monthly billing cycles, periods spanning more than 35 days should be dropped from analysis. 
    For bi-monthly billing cycles, periods spanning more than 70 days should be dropped from the analysis.
    """

    def __init__(self, data : pd.DataFrame, is_electricity_data : bool):
        self._df = None
        self.warnings = []
        self.disqualification = []
        self.is_electricity_data = is_electricity_data

        self.set_data(data = data)
        for warning in self.warnings + self.disqualification:
            warning.warn()

    @property
    def df(self):
        if self._df is None:
            return None
        else:
            return self._df.copy()


    def _compute_meter_value_df(self, df : pd.DataFrame):
        # TODO : Assume if the billing cycle is mixed between monthly and bimonthly, then the minimum granularity is bimonthly
        # Test for more than 50% of high frequency data being missing
        """
            2.2.2.1. If summing to daily usage from higher frequency interval data, no more than 50% of high-frequency values should be missing. 
            Missing values should be filled in with average of non-missing values (e.g., for hourly data, 24 * average hourly usage).
        """
        meter_series = df['observed'].dropna()
        min_granularity = compute_minimum_granularity(meter_series.index, default_granularity='billing_bimonthly')

        # Ensure higher frequency data is aggregated to the monthly model
        if not min_granularity.startswith('billing'):
            min_granularity = 'billing_monthly'

        # This checks for offcycle reads. That is a disqualification if the billing cycle is less than 25 days
        meter_value_df = clean_caltrack_billing_daily_data(meter_series.to_frame('value'), min_granularity, self.disqualification)
        meter_value_df = meter_value_df.rename(columns={'value': 'observed'})
        # This will ensure that the missing days are kept in the dataframe
        # Create an index with all the days from the start and end date of 'meter_value_df'
        all_days_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D', tz=df.index.tz)
        all_days_df = pd.DataFrame(index=all_days_index)
        meter_value_df = meter_value_df.merge(all_days_df, left_index=True, right_index=True, how='outer')

        """
            Data Backfilling logic - 
            Reasoning is that the day we get the meter data is for the last month. So we should backfill it rather than forward fill it.
            To handle that the data at the end of the cycle is filled, we concatenate the first (filled) row at the end of the DataFrame and then backfill it.
            TODO : Verify if this is the right way in case there are actual missing values.
        """

        # Concatenate the first row to the end of the DataFrame
        meter_value_df = pd.concat([meter_value_df, meter_value_df.dropna().head(1)])
        meter_value_df = meter_value_df.bfill()
        meter_value_df = meter_value_df.iloc[:-1]

        return meter_value_df

    def _compute_temperature_features(self, df : pd.DataFrame, meter_value_df : pd.DataFrame):
        temp_series = df['temperature']
        temp_series.index.freq = temp_series.index.inferred_freq
        if temp_series.index.freq != 'H':
            if temp_series.index.freq is None or isinstance(temp_series.index.freq, MonthEnd) or temp_series.index.freq > pd.Timedelta(hours=1):
                # Add warning for frequencies longer than 1 hour
                self.warnings.append(
                    EEMeterWarning(
                        qualified_name="eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency",
                        description=("Cannot confirm that pre-aggregated temperature data had sufficient hours kept"),
                        data={},
                    )
                )
            # TODO consider disallowing this until a later patch
            if temp_series.index.freq != 'D':
                # Downsample / Upsample the temperature data to daily
                temperature_features = as_freq(temp_series, 'D', series_type = 'instantaneous', include_coverage = True)
                # If high frequency data check for 50% data coverage in rollup
                if len(temperature_features[temperature_features.coverage <= 0.5]) > 0:
                    self.warnings.append(
                        EEMeterWarning(
                            qualified_name="eemeter.caltrack_sufficiency_criteria.missing_high_frequency_temperature_data",
                            description=("More than 50% of the high frequency Temperature data is missing."),
                            data = (
                                temperature_features[temperature_features.coverage <= 0.5].index.to_list()
                            )
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
            temperature_features = compute_temperature_features(
                meter_value_df.index,
                temp_series,
                data_quality=True,
            )

        return temperature_features


    def _check_data_sufficiency(self, df : pd.DataFrame):
        
        meter_value_df = self._compute_meter_value_df(df)
        temperature_features = self._compute_temperature_features(df, meter_value_df)

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

    def set_data(self, data : pd.DataFrame):
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
        expected_columns = ["observed", "temperature"]
        if not set(expected_columns).issubset(set(data.columns)):
            # show the columns that are missing

            raise ValueError("Data is missing required columns: {}".format(
                set(expected_columns) - set(data.columns)))

        # Copy the input dataframe so that the original is not modified
        df = data.copy()

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
        elif str(df.index.tz) == 'UTC':
            warnings.warn('Datetime index is in UTC. Use tz_localize() with the local timezone to ensure correct aggregations')

        if self.is_electricity_data:
            df.loc[df['observed'] == 0, 'observed'] = np.nan

        # Data Sufficiency Check
        df = self._check_data_sufficiency(df)

        self._df = df


class BillingReportingData(AbstractDataProcessor):
    def __init__(self, data : pd.DataFrame, is_electricity_data : bool):
        self._df = None
        self.warnings = []
        self.disqualification = []
        self.is_electricity_data = is_electricity_data

        # TODO : do we need to set electric data for reporting?
        self.set_data(data = data)

    @property
    def df(self):
        if self._df is None:
            return None
        else:
            return self._df.copy()


    def _check_data_sufficiency(self, df : pd.DataFrame):

        temp_series = df['temperature']
        temp_series.index.freq = temp_series.index.inferred_freq
        if temp_series.index.freq is None or isinstance(temp_series.index.freq, MonthEnd) or temp_series.index.freq > pd.Timedelta(hours=1):
                # Add warning for frequencies longer than 1 hour
                self.warnings.append(
                    EEMeterWarning(
                        qualified_name="eemeter.caltrack_sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency",
                        description=("Cannot confirm that pre-aggregated temperature data had sufficient hours kept"),
                        data={},
                    )
                )
        if temp_series.index.freq != 'D':
            temperature_df = as_freq(temp_series, 'D', series_type = 'instantaneous', include_coverage=True)
            # If high frequency data check for 50% data coverage in rollup
            if len(temperature_df[temperature_df.coverage <= 0.5]) > 0:
                self.warnings.append(
                    EEMeterWarning(
                        qualified_name="eemeter.caltrack_sufficiency_criteria.missing_high_frequency_temperature_data",
                        description=("More than 50% of the high frequency Temperature data is missing."),
                        data = (
                            temperature_df[temperature_df.coverage <= 0.5].index.to_list()
                        )
                    )
                )

            # Set missing high frequency data to NaN
            temperature_df.value[temperature_df.coverage > 0.5] = (
                temperature_df[temperature_df.coverage > 0.5].value / temperature_df[temperature_df.coverage > 0.5].coverage
            )

            temperature_df = temperature_df[temperature_df.coverage > 0.5].reindex(temperature_df.index)[["value"]].rename(columns={'value' : 'temperature'})
            
            if 'coverage' in temperature_df.columns:
                temperature_df = temperature_df.drop(columns=['coverage'])
        else :
            temperature_df = temp_series.to_frame(name='temperature')

        # This will ensure that the missing days are kept in the dataframe
        # Create an index with all the days from the start and end date of 'meter_value_df'
        all_days_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D', tz=df.index.tz)
        all_days_df = pd.DataFrame(index=all_days_index)
        temperature_df = temperature_df.merge(all_days_df, left_index=True, right_index=True, how='outer')

        temperature_df['temperature_null'] = temperature_df.temperature.isnull().astype(int)
        temperature_df['temperature_not_null'] = temperature_df.temperature.notnull().astype(int)

        temperature_df, self.disqualification, warnings = caltrack_sufficiency_criteria_baseline(data = temperature_df, is_reporting_data = True)

        self.warnings += warnings

        # drop data quality columns
        temperature_df = temperature_df.drop(columns=['temperature_null', 'temperature_not_null'])

        # TODO : interpolate if necessary

        return temperature_df

    def set_data(self, data : pd.DataFrame):
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

        self._df = df