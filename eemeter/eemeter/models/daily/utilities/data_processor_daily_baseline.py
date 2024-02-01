from eemeter.common.abstract_data_processor import AbstractDataProcessor
import eemeter.common.const as _const
from config import DailySettings
from eemeter.eemeter.common.data_processor_utilities import (
    day_counts,
    clean_caltrack_billing_daily_data
)
from eemeter.eemeter.warnings import EEMeterWarning
import numpy as np
import pandas as pd
import pytz

class DataProcessorDailyBaseline(AbstractDataProcessor):
    """Data processor for daily data.
    
    2.2.1.4. Values of 0 are considered missing for electricity data, but not gas data.
    
    """
    def __init__(self, settings : DailySettings | None):
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
        self._meter_id = None
        self._sufficiency_warnings = None 

    def _caltrack_sufficiency_criteria_baseline(self,
        data_quality,
        requested_start = None,
        requested_end = None,
        num_days=365,
        min_fraction_daily_coverage=0.9,
        min_fraction_hourly_temperature_coverage_per_period=0.9,
    ):
        """
            Refer to usage_per_day.py in eemeter/caltrack/ folder
        """

        if data_quality.dropna().empty:
            warnings.append(
                    EEMeterWarning(
                        qualified_name="eemeter.caltrack_sufficiency_criteria.no_data",
                        description=("No data available."),
                        data={},
                    )
            )
            return warnings

        data_start = data_quality.index.min().tz_convert("UTC")
        data_end = data_quality.index.max().tz_convert("UTC")
        n_days_data = (data_end - data_start).days

        if requested_start is not None:
            # check for gap at beginning
            requested_start = requested_start.astimezone(pytz.UTC)
            n_days_start_gap = (data_start - requested_start).days
        else:
            n_days_start_gap = 0

        if requested_end is not None:
            # check for gap at end
            requested_end = requested_end.astimezone(pytz.UTC)
            n_days_end_gap = (requested_end - data_end).days
        else:
            n_days_end_gap = 0

        critical_warnings = []

        if n_days_end_gap < 0:
            # CalTRACK 2.2.4
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria"
                        ".extra_data_after_requested_end_date"
                    ),
                    description=("Extra data found after requested end date."),
                    data={
                        "requested_end": requested_end.isoformat(),
                        "data_end": data_end.isoformat(),
                    },
                )
            )
            n_days_end_gap = 0

        if n_days_start_gap < 0:
            # CalTRACK 2.2.4
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria"
                        ".extra_data_before_requested_start_date"
                    ),
                    description=("Extra data found before requested start date."),
                    data={
                        "requested_start": requested_start.isoformat(),
                        "data_start": data_start.isoformat(),
                    },
                )
            )
            n_days_start_gap = 0

        n_days_total = n_days_data + n_days_start_gap + n_days_end_gap

        n_negative_meter_values = data_quality.meter_value[
            data_quality.meter_value < 0
        ].shape[0]

        if n_negative_meter_values > 0:
            # CalTrack 2.3.5
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria" ".negative_meter_values"
                    ),
                    description=(
                        "Found negative meter data values, which may indicate presence"
                        " of solar net metering."
                    ),
                    data={"n_negative_meter_values": n_negative_meter_values},
                )
            )

        # TODO(philngo): detect and report unsorted or repeated values.

        # create masks showing which daily or billing periods meet criteria

        # TODO : How to handle temperature if already rolled up in the dataframe?
        valid_meter_value_rows = data_quality.meter_value.notnull()
        valid_temperature_rows = (
            data_quality.temperature_not_null
            / (data_quality.temperature_not_null + data_quality.temperature_null)
        ) > min_fraction_hourly_temperature_coverage_per_period
        valid_rows = valid_meter_value_rows & valid_temperature_rows

        # get number of days per period - for daily this should be a series of ones
        row_day_counts = day_counts(data_quality.index)

        # apply masks, giving total
        n_valid_meter_value_days = int((valid_meter_value_rows * row_day_counts).sum())
        n_valid_temperature_days = int((valid_temperature_rows * row_day_counts).sum())
        n_valid_days = int((valid_rows * row_day_counts).sum())

        median = data_quality.meter_value.median()
        upper_quantile = data_quality.meter_value.quantile(0.75)
        lower_quantile = data_quality.meter_value.quantile(0.25)
        iqr = upper_quantile - lower_quantile
        extreme_value_limit = median + (3 * iqr)
        n_extreme_values = data_quality.meter_value[
            data_quality.meter_value > extreme_value_limit
        ].shape[0]
        max_value = float(data_quality.meter_value.max())

        if n_days_total > 0:
            fraction_valid_meter_value_days = n_valid_meter_value_days / float(n_days_total)
            fraction_valid_temperature_days = n_valid_temperature_days / float(n_days_total)
            fraction_valid_days = n_valid_days / float(n_days_total)
        else:
            # unreachable, I think.
            fraction_valid_meter_value_days = 0
            fraction_valid_temperature_days = 0
            fraction_valid_days = 0

        if n_days_total != num_days:
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria"
                        ".incorrect_number_of_total_days"
                    ),
                    description=("Total data span does not match the required value."),
                    data={"num_days": num_days, "n_days_total": n_days_total},
                )
            )

        if fraction_valid_days < min_fraction_daily_coverage:
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria"
                        ".too_many_days_with_missing_data"
                    ),
                    description=(
                        "Too many days in data have missing meter data or"
                        " temperature data."
                    ),
                    data={"n_valid_days": n_valid_days, "n_days_total": n_days_total},
                )
            )

        if fraction_valid_meter_value_days < min_fraction_daily_coverage:
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria"
                        ".too_many_days_with_missing_meter_data"
                    ),
                    description=("Too many days in data have missing meter data."),
                    data={
                        "n_valid_meter_data_days": n_valid_meter_value_days,
                        "n_days_total": n_days_total,
                    },
                )
            )


        if fraction_valid_temperature_days < min_fraction_daily_coverage:
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria"
                        ".too_many_days_with_missing_temperature_data"
                    ),
                    description=("Too many days in data have missing temperature data."),
                    data={
                        "n_valid_temperature_data_days": n_valid_temperature_days,
                        "n_days_total": n_days_total,
                    },
                )
            )


        # Test for more than 50% of high frequency data being missing
        """
            2.2.2.1. If summing to daily usage from higher frequency interval data, no more than 50% of high-frequency values should be missing. 
            Missing values should be filled in with average of non-missing values (e.g., for hourly data, 24 * average hourly usage).
        """  
        # TODO : Figure out minimum granularity
        min_granularity = 'unknown'
        if min_granularity != 'daily':
            # TODO: the warning applied should be specific to temperature or meter data
            data_quality['meter_value'] = clean_caltrack_billing_daily_data(data_quality['meter_value'], min_granularity, critical_warnings)
            data_quality['temperature_mean'] = clean_caltrack_billing_daily_data(data_quality['temperature_mean'], min_granularity, critical_warnings)

        
        # Check for 90% for individual months present:
        non_null_temp_percentage_per_month = data_quality['temperature_mean'].groupby(data_quality.index.month).apply(lambda x: x.notna().mean())
        if (non_null_temp_percentage_per_month < min_fraction_daily_coverage).any():
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.caltrack_sufficiency_criteria.missing_temperature_data",
                    description=("More than 10% of the monthly temperature data is missing."),

                )
            )

        non_null_meter_percentage_per_month = data_quality['meter_value'].groupby(data_quality.index.month).apply(lambda x: x.notna().mean())
        if (non_null_meter_percentage_per_month < min_fraction_daily_coverage).any():
            critical_warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.caltrack_sufficiency_criteria.missing_meter_data",
                    description=("More than 10% of the monthly meter data is missing."),

                )
            )
        
        # TODO : Do we need 90% of seasons & weekday/weekend checks needed?

        non_critical_warnings = []
        if n_extreme_values > 0:
            # CalTRACK 2.3.6
            non_critical_warnings.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.caltrack_sufficiency_criteria" ".extreme_values_detected"
                    ),
                    description=(
                        "Extreme values (greater than (median + (3 * IQR)),"
                        " must be flagged for manual review."
                    ),
                    data={
                        "n_extreme_values": n_extreme_values,
                        "median": median,
                        "upper_quantile": upper_quantile,
                        "lower_quantile": lower_quantile,
                        "extreme_value_limit": extreme_value_limit,
                        "max_value": max_value,
                    },
                )
            )

        warnings = critical_warnings + non_critical_warnings

        return warnings

    def _check_data_sufficiency(self, df : pd.DataFrame):
        """
            https://docs.caltrack.org/en/latest/methods.html#section-2-data-management
            Check under section 2.2 : Data constraints
        """

        # Add Season and Weekday_weekend columns
        df['season'] = df.index.month_name().map(_const.default_season_def)
        df['weekday_weekend'] = df.index.day_name().map(_const.default_weekday_weekend_def)

        df['temperature_null'] = df.temperature_mean.isnull().astype(int)
        df['temperature_not_null'] = df.temperature_mean.notnull().astype(int)

        warnings = self._caltrack_sufficiency_criteria_baseline(df)

        # Drop unnecessary columns
        df.drop(columns=['temperature_null', 'temperature_not_null'], inplace=True)

        self._sufficiency_warnings = warnings

        # TODO : Add interpolation if the data is correct

    def _interpolate_data(self, df : pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing data in the dataframe
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with missing data
        
        Returns
        -------
        pd.DataFrame
            Dataframe with missing data interpolated
        """
        # TODO : Is this actually necessary? Or just throw a warning if we have missing data?
        return df
    
    
    def set_data(self, data : pd.DataFrame, is_electricity_data : bool):
        """Process data input for the Daily Model Baseline Class
        Assumes that datetime is already index.
        
        Parameters
        ----------
        data : pd.DataFrame
            Required columns - date time, meter value, temperature mean
        
        Returns
        -------
        processed_data : pd.DataFrame
            Dataframe appended with the correct season and day of week.
        """

        expected_columns = ["meter_value", "temperature_mean"]
        if not set(expected_columns).issubset(set(data.columns)):
            # show the columns that are missing

            raise ValueError("Data is missing required columns: {}".format(
                set(expected_columns) - set(data.columns)))

        # Handle the case if the datetime is not the index but provided in a separate column -> not required for now

        # Check that the datetime index is timezone aware timestamp
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Index is not datetime")
        elif data.index.tz is None:
            raise ValueError("Datatime is missing timezone information")
        

        # Copy the input dataframe so that the original is not modified
        df = data.copy()

        # TODO : Check missing value in datetime and add a warning/ exception if missing

        # Convert electricity data having 0 meter values to NaNs
        if is_electricity_data:
            df.loc[df['meter_value'] == 0, 'meter_value'] = np.nan


        # Data Sufficiency Check
        self._check_data_sufficiency(df)
    
        if self._sufficiency_warnings is not None:
            for warning in self._sufficiency_warnings:
                print(warning.json())

        # Is this the valid way for doing the rollups?
        df['meter_value'] = df['meter_value'] / day_counts(df.index)
        self._baseline_meter_df = df


if __name__ == "__main__":

    data = pd.read_csv("eemeter/common/test_data.csv")
    data.drop(columns=['season', 'day_of_week'], inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data.set_index('datetime', inplace=True)

    # print(data.head())

    cl = DataProcessorDailyBaseline(None)

    cl.set_data(data, is_electricity_data=True)

    print(cl._baseline_meter_df.head())
    print(cl._sufficiency_warnings)

