from eemeter.common.abstract_data_processor import AbstractDataProcessor
from eemeter.common.data_settings import MonthlySettings
from eemeter.eemeter.common.data_processor_utilities import (
    caltrack_sufficiency_criteria_baseline,
    compute_minimum_granularity,
    as_freq
)
import pandas as pd

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
        self.sufficiency_warnings = None 
        self.critical_sufficiency_warnings = None

        # TODO : do we need to set electric data for reporting?
        self.set_data(data = data, is_electricity_data = False)


    def _check_data_sufficiency(self, df : pd.DataFrame):

        df['temperature_null'] = df.temperature_mean.isnull().astype(int)
        df['temperature_not_null'] = df.temperature_mean.notnull().astype(int)

        df, self.critical_sufficiency_warnings, self.sufficiency_warnings = caltrack_sufficiency_criteria_baseline(data = df, is_reporting_data = True)

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

        if self.critical_sufficiency_warnings or self.sufficiency_warnings:
            for warning in self.critical_sufficiency_warnings + self.sufficiency_warnings:
                print(warning.json())

        self._reporting_meter_df = df

if __name__ == "__main__":

    data = pd.read_csv("eemeter/common/test_data.csv")
    data.drop(columns=['season', 'day_of_week', 'meter_value'], inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data.set_index('datetime', inplace=True)

    print(data.head())

    cl = DataBillingReporting(data = data)

    print(cl._reporting_meter_df.head())