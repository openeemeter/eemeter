import pandas as pd
from pandas.tseries.frequencies import to_offset


class ModelDataFormatter(object):
    ''' Formatter for model data of known or predictable frequency.
    Basic usage:

    .. code-block:: python

        >>> formatter = ModelDataFormatter("D")
        >>> formatter.create_input(energy_trace, weather_source)
                                   energy tempF
        2013-06-01 00:00:00+00:00    3.10  74.3
        2013-06-02 00:00:00+00:00    2.42  71.0
        2013-06-03 00:00:00+00:00    1.38  73.1
                                           ...
        2016-05-27 00:00:00+00:00    0.11  71.1
        2016-05-28 00:00:00+00:00    0.04  78.1
        2016-05-29 00:00:00+00:00    0.21  69.6
        >>> index = pd.date_range('2013-01-01', periods=365, freq='D')
        >>> formatter.create_input(index, weather_source)
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

    def __repr__(self):
        return 'ModelDataFormatter("{}")'.format(self.freq_str)

    def create_input(self, trace, weather_source):
        '''Creates a :code:`DatetimeIndex` ed dataframe containing formatted
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
            usable as input to applicable model.fit() methods.
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
        '''Creates a :code:`DatetimeIndex` ed dataframe containing formatted
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


class ModelDataBillingFormatter():
    ''' Formatter for model data of unknown or unpredictable frequency.
    Basic usage:

    .. code-block:: python

        >>> formatter = ModelDataBillingFormatter()
        >>> energy_trace = EnergyTrace(
                "ELECTRICITY_CONSUMPTION_SUPPLIED",
                pd.DataFrame(
                    {
                        "value": [1, 1, 1, 1, np.nan],
                        "estimated": [False, False, True, False, False]
                    },
                    index=[
                        datetime(2011, 1, 1, tzinfo=pytz.UTC),
                        datetime(2011, 2, 1, tzinfo=pytz.UTC),
                        datetime(2011, 3, 2, tzinfo=pytz.UTC),
                        datetime(2011, 4, 3, tzinfo=pytz.UTC),
                        datetime(2011, 4, 29, tzinfo=pytz.UTC),
                    ],
                    columns=["value", "estimated"]
                ),
                unit="KWH")
        >>> trace_data, temp_data = \
formatter.create_input(energy_trace, weather_source)
        >>> trace_data
        2011-01-01 00:00:00+00:00    1.0
        2011-02-01 00:00:00+00:00    1.0
        2011-03-02 00:00:00+00:00    2.0
        2011-04-29 00:00:00+00:00    NaN
        dtype: float64
        >>> temp_data
        period                    hourly
        2011-01-01 00:00:00+00:00 2011-01-01 00:00:00+00:00  32.0
                                  2011-01-01 01:00:00+00:00  32.0
                                  2011-01-01 02:00:00+00:00  32.0
        ...                                                   ...
        2011-03-02 00:00:00+00:00 2011-04-28 21:00:00+00:00  32.0
                                  2011-04-28 22:00:00+00:00  32.0
                                  2011-04-28 23:00:00+00:00  32.0
        >>> index = pd.date_range('2013-01-01', periods=365, freq='D')
        >>> formatter.create_input(index, weather_source)
                                   tempF
        2013-01-01 00:00:00+00:00   28.3
        2013-01-02 00:00:00+00:00   31.0
        2013-01-03 00:00:00+00:00   34.1
                                    ...
        2013-12-29 00:00:00+00:00   12.3
        2013-12-30 00:00:00+00:00   26.0
        2013-12-31 00:00:00+00:00   24.1
    '''

    def __repr__(self):
        return 'ModelDataBillingFormatter()'

    def _unestimated(self, data):
        def _yield_values():
            index, value = None, None
            for i, row in data.iterrows():
                if row.estimated:
                    if index is None:
                        index, value = i, row.value
                    else:
                        value += row.value
                else:
                    if index is None:
                        yield i, row.value
                    else:
                        yield index, (value + row.value)
                    index, value = None, None

        yielded = list(_yield_values())
        if len(yielded) == 0:
            return pd.Series()
        index, values = zip(*yielded)
        return pd.Series(values, index=index)

    def create_input(self, trace, weather_source):
        '''Creates two :code:`DatetimeIndex` ed dataframes containing formatted
        model input data formatted as follows.

        Parameters
        ----------
        trace : eemeter.structures.EnergyTrace
            The source of energy data for inclusion in model input.
        weather_source : eemeter.weather.WeatherSourceBase
            The source of weather data.

        Returns
        -------
        trace_data : pandas.DataFrame
            Predictably formatted trace data with estimated data removed.
            This data should be directly usable as input to applicable
            model.fit() methods.
        temperature_data : pandas.DataFrame
            Predictably formatted temperature data with a pandas
            :code:`MultiIndex`.  The :code:`MultiIndex` contains two levels
            - 'period', which corresponds directly to the trace_data index,
            and 'hourly' or 'daily', which contains, respectively, hourly or
            daily temperature data. This is intended for use like the
            following:

            .. code-block:: python

                >>> temperature_data.groupby(level='period')

            This data should be directly usable as input to applicable
            model.fit() methods.
        '''
        unestimated_trace_data = self._unestimated(trace.data.copy())
        temp_data = weather_source.indexed_temperatures(
            unestimated_trace_data.index, "degF", allow_mixed_frequency=True)
        return unestimated_trace_data, temp_data

    def create_demand_fixture(self, index, weather_source):
        '''Creates a :code:`DatetimeIndex` ed dataframe containing formatted
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
