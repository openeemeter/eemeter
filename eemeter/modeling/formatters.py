'''
These take usage data or a pandas index and attach weather data
'''

from collections import OrderedDict

import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset


class FormatterBase(object):

    def create_input(self, trace, weather_source):
        message = (
            "Inheriting classes must implement the `create_input(` method."
        )
        raise NotImplementedError(message)

    def create_demand_fixture(self, index, weather_source):
        message = (
            'Inheriting classes must implement the'
            ' `create_demand_fixture(` method.'
        )
        raise NotImplementedError(message)

    def _get_start_date(self, input_data):
        return None

    def _get_end_date(self, input_data):
        return None

    def _get_n_rows(self, input_data):
        return 0

    def describe_input(self, input_data):
        ''' Describes input data in a consistent format.

        Parameters
        ----------
        input_data : pandas.DataFrame
            input_data as given by `self.create_input(trace, weather_source)`

        Returns
        -------
        description : dict

            - **start_date**: earliest date of input data.
            - **end_date**: latest date of input data.
            - **n_rows**: number of rows of data.
        '''
        return {
            "start_date": self._get_start_date(input_data),
            "end_date": self._get_end_date(input_data),
            "n_rows": self._get_n_rows(input_data),
        }


class ModelDataFormatter(FormatterBase):
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

    def _get_start_date(self, input_data):
        if input_data.shape[0] >= 1:
            return input_data.index[0]
        return None

    def _get_end_date(self, input_data):
        if input_data.shape[0] >= 1:
            return input_data.index[-1]
        return None

    def _get_n_rows(self, input_data):
        return input_data.shape[0]

    def serialize_input(self, input_data):
        ''' Serialize input data
        '''
        return OrderedDict([
            (start.isoformat(), OrderedDict([
                ("energy", row.energy if pd.notnull(row.energy) else None),
                ("tempF", row.tempF if pd.notnull(row.tempF) else None),
            ]))
            for start, row in input_data.iterrows()
        ])

    def serialize_demand_fixture(self, demand_fixture_data):
        ''' Serialize demand fixture data
        '''
        return OrderedDict([
            (i.isoformat(), row.tempF)
            for i, row in demand_fixture_data.iterrows()
        ])

    def get_input_data_mask(self, input_data):
        ''' Boolean list of missing/not missing values:
            True  => missing
            False => not missing
        '''

        dts, mask = [], []
        for i, row in input_data.iterrows():
            dts.append(i)
            mask.append(pd.isnull(row.energy) or pd.isnull(row.tempF))
        return pd.Series(mask, index=dts)

    def daily_trace_data(self, trace):
        ''' Transforms a trace for this formatter to a daily series
        '''
        return trace.data.value.resample('D').sum()  # discard 'estimated' col


class ModelDataBillingFormatter(FormatterBase):
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

    def _get_start_date(self, input_data):
        unestimated_trace_data, temp_data = input_data
        if unestimated_trace_data.shape[0] >= 1:
            return unestimated_trace_data.index[0]
        return None

    def _get_end_date(self, input_data):
        unestimated_trace_data, temp_data = input_data
        if unestimated_trace_data.shape[0] >= 1:
            return unestimated_trace_data.index[-1]
        return None

    def _get_n_rows(self, input_data):
        unestimated_trace_data, temp_data = input_data
        return unestimated_trace_data.shape[0]

    def serialize_input(self, input_data):
        trace_data, temp_data = input_data

        # must be careful because empty does not carry multiindex
        if trace_data.shape[0] == 0:
            return OrderedDict([])

        # funky stuff in here manages the multiindex on the temperature data
        return OrderedDict([
            (start.isoformat(), OrderedDict([
                ("energy", energy),
                ("tempF", OrderedDict([
                    (i[1].isoformat(), v[0]) for i, v in group.iterrows()
                ])),
            ]))
            for (start, energy), (p, group) in
            zip(trace_data.iteritems(), temp_data.groupby(level="period"))
        ])

    def serialize_demand_fixture(self, demand_fixture_data):
        return OrderedDict([
            (i.isoformat(), row.tempF)
            for i, row in demand_fixture_data.iterrows()
        ])

    def get_input_data_mask(self, input_data):
        ''' Boolean list of missing/not missing values:
            True  => missing
            False => not missing
        '''
        trace_data, temp_data = input_data
        dts = []
        mask = []
        if trace_data.empty or temp_data.empty:
            return pd.Series(mask)
        for (start, energy), (p, group) in zip(
                trace_data.iteritems(),
                temp_data.groupby(level="period")):
            temps = group.copy()
            temps.index = temps.index.droplevel()
            daily_temps = temps.resample('D').apply(np.mean)[0]
            for i, tempF in daily_temps.iteritems():
                dts.append(i)
                mask.append(pd.isnull(energy) or pd.isnull(tempF))
        return pd.Series(mask, index=dts)

    def daily_trace_data(self, trace):
        ''' Transforms a trace for this formatter to a daily series
        '''
        if trace.data.empty:
            return pd.Series([])
        data = [
            value / (e - s).days for value, s, e in
            zip(trace.data.value, trace.data.index, trace.data.index[1:])
        ] + [np.nan]  # add missing last data point

        retval = pd.Series(data, index=trace.data.index)
        retval = retval[
            ~retval.index.duplicated(keep='last')].sort_index()
        retval = retval.resample('D').ffill()
        return retval
