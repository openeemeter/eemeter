from eemeter.common.abstract_data_processor import AbstractDataProcessor
import pandas as pd

class DataProcessorDaily(AbstractDataProcessor):
    """Data processor for daily data."""
    def __init__(self, settings):
        """Initialize the data processor.
        
        Parameters
        ----------
        settings : DailySettings
            Settings for the data processor.
        """
        self.settings = settings

    def set_data(self, data : pd.DataFrame):
        """Process data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Required columns - date time, meter value, temperature mean, timezone
        
        Returns
        -------
        processed_data : pd.DataFrame
            Dataframe appended with the correct season and day of week.
        """
        pass
