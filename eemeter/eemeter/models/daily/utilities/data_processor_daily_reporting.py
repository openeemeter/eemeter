from eemeter.common.abstract_data_processor import AbstractDataProcessor
from eemeter.eemeter.transform import (
    clean_caltrack_billing_daily_data,
    get_reporting_data
)
import pandas as pd

class DataProcessorDailyReporting(AbstractDataProcessor):

    """
        Refer to Section 3.5 in https://docs.caltrack.org/en/latest/methods.html#section-2-data-management

        The Set data will be very similar (might be the exact same) as the Baseline version of this class. The only difference will be
        the data_sufficiency check. Although that will also be reused.
    """
    
    def _check_data_sufficiency(self, data):
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
        min_granularity = 'unknown'
        data['meter_value'] = clean_caltrack_billing_daily_data(data['meter_value'], min_granularity)  
        
              
        pass

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

        # TODO : try and abstract out the baseline version instead of copying it here

        pass