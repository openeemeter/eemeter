from eemeter.common.abstract_data_processor import AbstractDataProcessor
import eemeter.common.const as _const
from config import DailySettings
from eemeter.eemeter.common.data_processor_utilities import (
    caltrack_sufficiency_criteria_baseline,
    clean_caltrack_billing_daily_data,
    compute_minimum_granularity,
    day_counts,
    as_freq
)
from eemeter.eemeter.warnings import EEMeterWarning
import numpy as np
import pandas as pd

class DataProcessorDailyBaseline(AbstractDataProcessor):
    """Data processor for daily data.
    
    2.2.1.4. Values of 0 are considered missing for electricity data, but not gas data.
    
    """
    def __init__(self, data : pd.DataFrame, is_electricity_data : bool, settings : DailySettings | None = None):
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
        self.sufficiency_warnings = None 
        self.critical_sufficiency_warnings = None

        self.set_data(data = data, is_electricity_data = is_electricity_data)


    def _check_data_sufficiency(self, df : pd.DataFrame):
        """
            https://docs.caltrack.org/en/latest/methods.html#section-2-data-management
            Check under section 2.2 : Data constraints
        """

        # Add Season and Weekday_weekend 
        df['season'] = df.index.month_name().map(_const.default_season_def)
        df['weekday_weekend'] = df.index.day_name().map(_const.default_weekday_weekend_def)

        df['temperature_null'] = df.temperature_mean.isnull().astype(int)
        df['temperature_not_null'] = df.temperature_mean.notnull().astype(int)

        df, self.critical_sufficiency_warnings, self.sufficiency_warnings = caltrack_sufficiency_criteria_baseline(data = df)


        # Test for more than 50% of high frequency data being missing
        """
            2.2.2.1. If summing to daily usage from higher frequency interval data, no more than 50% of high-frequency values should be missing. 
            Missing values should be filled in with average of non-missing values (e.g., for hourly data, 24 * average hourly usage).
        """  
        min_granularity = compute_minimum_granularity(df.index)

        if min_granularity.startswith('billing'):
            raise ValueError("Billing data is not allowed in the daily model")
            
        # Ensure higher frequency data is aggregated to the monthly model
        meter_value_df = clean_caltrack_billing_daily_data(df['meter_value'], min_granularity, self.sufficiency_warnings)
        meter_value_df.rename(columns={'value': 'meter_value'}, inplace=True)

        temperature_df = as_freq(df['temperature_mean'], 'D', series_type = 'instantaneous').to_frame(name='temperature_mean')

        # Perform a join
        meter_value_df = meter_value_df.merge(temperature_df, left_index=True, right_index=True, how='outer')

        df = meter_value_df
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

        # Convert electricity data having 0 meter values to NaNs
        if is_electricity_data:
            df.loc[df['meter_value'] == 0, 'meter_value'] = np.nan


        # Data Sufficiency Check
        df = self._check_data_sufficiency(df)

        # TODO : how to handle the warnings? Should we throw an exception or just print the warnings?
        if self.critical_sufficiency_warnings or self.sufficiency_warnings:
            for warning in self.critical_sufficiency_warnings + self.sufficiency_warnings:
                print(warning.json())

        # TODO : Resample high frequency data into daily data
        df['meter_value'] = df['meter_value'] / day_counts(df.index)
        df = df.resample('D').mean()
        self._baseline_meter_df = df


if __name__ == "__main__":

    data = pd.read_csv("eemeter/common/test_data.csv")
    data.drop(columns=['season', 'day_of_week'], inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data.set_index('datetime', inplace=True)

    print(data.head())

    cl = DataProcessorDailyBaseline(data = data, is_electricity_data=True)

    print(cl._baseline_meter_df.head())

