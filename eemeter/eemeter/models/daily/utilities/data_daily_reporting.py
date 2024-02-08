from config import DailySettings
from eemeter.common.abstract_data_processor import AbstractDataProcessor
from eemeter.eemeter.common.data_processor_utilities import (
    caltrack_sufficiency_criteria_baseline,
    compute_minimum_granularity,
    as_freq
)
import pandas as pd

class DataDailyReporting(AbstractDataProcessor):

    """
        Refer to Section 3.5 in https://docs.caltrack.org/en/latest/methods.html#section-2-data-management

        The Set data will be very similar (might be the exact same) as the Baseline version of this class. The only difference will be
        the data_sufficiency check. Although that will also be reused.
    """
    def __init__(self, data : pd.DataFrame, settings : DailySettings | None = None):
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
        df['temperature_null'] = df.temperature_mean.isnull().astype(int)
        df['temperature_not_null'] = df.temperature_mean.notnull().astype(int)

        df, self.disqualification, self.warnings = caltrack_sufficiency_criteria_baseline(data = df, is_reporting_data = True)

        df = as_freq(df['temperature_mean'], 'D', series_type = 'instantaneous').to_frame(name='temperature_mean') 
        
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

if __name__ == "__main__":

    data = pd.read_csv("eemeter/common/test_data.csv")
    data.drop(columns=['season', 'day_of_week', 'meter_value'], inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)
    data.set_index('datetime', inplace=True)

    print(data.head())

    cl = DataDailyReporting(data = data)

    print(cl._reporting_meter_df.head())