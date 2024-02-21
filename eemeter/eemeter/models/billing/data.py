from eemeter.eemeter.common.data_processor_utilities import as_freq, caltrack_sufficiency_criteria_baseline, clean_caltrack_billing_daily_data, compute_minimum_granularity
from eemeter import compute_temperature_features
from eemeter.eemeter.warnings import EEMeterWarning
from eemeter.eemeter.models.daily.data import _DailyData

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd, MonthBegin

from typing import Optional, Union


"""TODO there is still a ton of unecessarily duplicated code between billing+daily.
    we should be able to perform a few transforms within the billing baseclass, and then call super() for the rest

    unsure whether we should inherit from the public classes because we'll have to take care to use type(data)
    instead of isinstance(data,  _) when doing the checks in the model/wrapper to avoid unintentionally allowing a mix of data/model type
"""

class _BillingData(_DailyData):
    """Baseline data processor for billing data.

    2.2.3.4. Off-cycle reads (spanning less than 25 days) should be dropped from analysis. 
    These readings typically occur due to meter reading problems or changes in occupancy.

    2.2.3.5. For pseudo-monthly billing cycles, periods spanning more than 35 days should be dropped from analysis. 
    For bi-monthly billing cycles, periods spanning more than 70 days should be dropped from the analysis.
    """

    def _compute_meter_value_df(self, df: pd.DataFrame):
        # TODO : Assume if the billing cycle is mixed between monthly and bimonthly, then the minimum granularity is bimonthly
        # Test for more than 50% of high frequency data being missing
        """
            2.2.2.1. If summing to daily usage from higher frequency interval data, no more than 50% of high-frequency values should be missing. 
            Missing values should be filled in with average of non-missing values (e.g., for hourly data, 24 * average hourly usage).
        """
        meter_series = df['observed'].dropna()
        min_granularity = compute_minimum_granularity(meter_series.index, default_granularity='billing_bimonthly')
        print(min_granularity)

        # Ensure higher frequency data is aggregated to the monthly model
        if not min_granularity.startswith('billing'):
            meter_series = meter_series.resample('MS').sum()
            self.warnings.append( 
                EEMeterWarning(
                        qualified_name="eemeter.caltrack_sufficiency_criteria.inferior_model_usage",
                        description=("Daily data is provided but the model used is monthly. Are you sure this is the intended model?"),
                        data={},
                    )
            )
            min_granularity = 'billing_monthly'

        # This checks for offcycle reads. That is a disqualification if the billing cycle is less than 25 days
        meter_value_df = clean_caltrack_billing_daily_data(meter_series.to_frame('value'), min_granularity, self.disqualification)
        
        # Spread billing data to daily
        if min_granularity.startswith('billing'):
            meter_value_df = as_freq(meter_value_df['value'], 'D').to_frame('value')

        meter_value_df = meter_value_df.rename(columns={'value': 'observed'})

        # Convert all non-zero time datetimes to zero time (i.e., set the time to midnight), for proper join since we only want one reading per day for billing
        meter_value_df.index = meter_value_df.index.normalize()

        # This will ensure that the missing days are kept in the dataframe
        # Create an index with all the days from the start and end date of 'meter_value_df'
        all_days_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D', tz=df.index.tz)
        all_days_df = pd.DataFrame(index=all_days_index)
        meter_value_df = meter_value_df.merge(all_days_df, left_index=True, right_index=True, how='outer')

        # Forward fill the data since it is assumed the meter date is for the start date of the billing cycle
        meter_value_df = meter_value_df.ffill()

        return meter_value_df

    def _compute_temperature_features(self, df: pd.DataFrame, meter_index: pd.DatetimeIndex):
        temp_series = df['temperature']
        temp_series.index.freq = temp_series.index.inferred_freq
        if temp_series.index.freq != 'H':
            if temp_series.index.freq is None or isinstance(temp_series.index.freq, MonthEnd) or isinstance(temp_series.index.freq, MonthBegin) or temp_series.index.freq > pd.Timedelta(hours=1):
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
                meter_index,
                temp_series,
                data_quality=True,
            )
        temp = temperature_features['temperature_mean'].rename('temperature')
        features = temperature_features.drop(columns=['temperature_mean'])
        return temp, features


class BillingBaselineData(_BillingData):
    def _check_data_sufficiency(self, sufficiency_df):
        _, disqualification, warnings = caltrack_sufficiency_criteria_baseline(sufficiency_df, is_reporting_data=False, is_electricity_data=self.is_electricity_data)
        return disqualification, warnings


class BillingReportingData(_BillingData):
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool):
        if 'observed' not in df.columns:
            df = df.copy()
            df['observed'] = np.nan
        super().__init__(df, is_electricity_data)        

    @classmethod
    def from_series(cls, meter_data: Optional[Union[pd.Series, pd.DataFrame]], temperature_data: Union[pd.Series, pd.DataFrame], is_electricity_data: Optional[bool]=None, tzinfo=None):
        if tzinfo and meter_data is not None:
            raise ValueError('When passing meter data to BillingReportingData, convert its DatetimeIndex to local timezone first; `tzinfo` param should only be used in the absence of reporting meter data.')
        if is_electricity_data is None and meter_data is not None:
            raise ValueError('Must specify is_electricity_data when passing meter data.')
        if meter_data is None:
            meter_data = pd.DataFrame({'observed': np.nan}, index=temperature_data.index)
            if tzinfo:
                meter_data = meter_data.tz_convert(tzinfo)
        return super().from_series(meter_data, temperature_data, is_electricity_data)

    def _check_data_sufficiency(self, sufficiency_df):
        _, disqualification, warnings = caltrack_sufficiency_criteria_baseline(sufficiency_df, is_reporting_data=True, is_electricity_data=self.is_electricity_data)
        return disqualification, warnings     
