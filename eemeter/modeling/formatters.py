import pandas as pd
from pandas.tseries.frequencies import to_offset


class ModelDataFormatter(object):
    ''' Formatter for model data. Basic usage:

    .. code-block:: python

        >>> model_data_formatter = ModelDataFormatter("D")
        >>> model_data_formatter.create_input(energy_trace, weather_source)
                                   energy tempF
        2013-06-01 00:00:00+00:00    3.10  74.3
        2013-06-02 00:00:00+00:00    2.42  71.0
        2013-06-03 00:00:00+00:00    1.38  73.1
                                           ...
        2016-05-27 00:00:00+00:00    0.11  71.1
        2016-05-28 00:00:00+00:00    0.04  78.1
        2016-05-29 00:00:00+00:00    0.21  69.6
        >>> index = pd.date_range('2013-01-01', periods=365, freq='D')
        >>> model_data_formatter.create_input(index, weather_source)
                                   tempF
        2013-01-01 00:00:00+00:00   28.3
        2013-01-02 00:00:00+00:00   31.0
        2013-01-03 00:00:00+00:00   34.1
                                    ...
        2013-12-29 00:00:00+00:00   12.3
        2013-12-30 00:00:00+00:00   26.0
        2013-12-31 00:00:00+00:00   24.1

    '''

    def __init__(self, freq_str):
        self.freq_str = freq_str

    def create_input(self, trace, weather_source):
        '''Creates a :code:`DatetimeIndex`ed dataframe containing formatted
        model input data formatted as follows.

        Parameters
        ----------
        trace : eemeter.structures.EnergyTrace
            The source of energy data for inclusion in model input.
        weather_source : eemeter.weather.WeatherSourceBase
            The source of weather data.

        Returns
        -------
        input_df : pandas.DataFrame
            Predictably formatted input data. This data should be directly
            usable as input to applicable model.predict() methods.
        '''
        if (trace.data.index.freq is not None and
                to_offset(trace.data.index.freq) > to_offset(self.freq_str)):
            raise ValueError(
                "Will not upsample '{}' to '{}'"
                .format(trace.data.index.freq, self.freq_str)
            )

        energy = trace.data.value.resample(self.freq_str).sum()
        tempF = weather_source.indexed_temperatures(energy.index, "degF")
        return pd.DataFrame({"energy": energy, "tempF": tempF},
                            columns=["energy", "tempF"])

    def create_demand_fixture(self, index, weather_source):
        '''Creates a :code:`DatetimeIndex`ed dataframe containing formatted
        demand fixture data.

        Parameters
        ----------
        index : pandas.DatetimeIndex
            The desired index for demand fixture data.
        weather_source : eemeter.weather.WeatherSourceBase
            The source of weather fixture data.

        Returns
        -------
        input_df : pandas.DataFrame
            Predictably formatted input data. This data should be directly
            usable as input to applicable model.predict() methods.
        '''
        tempF = weather_source.indexed_temperatures(index, "degF")
        return pd.DataFrame({"tempF": tempF})
