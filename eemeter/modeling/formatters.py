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
        return OrderedDict([
            (start.isoformat(), OrderedDict([
                ("energy", row.energy if pd.notnull(row.energy) else None),
                ("tempF", row.tempF if pd.notnull(row.tempF) else None),
            ]))
            for start, row in input_data.iterrows()
        ])

    def serialize_demand_fixture(self, demand_fixture_data):
        return OrderedDict([
            (i.isoformat(), row.tempF)
            for i, row in demand_fixture_data.iterrows()
        ])


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


class CaltrackFormatter(FormatterBase):
    '''
    This formatter implements the data prep methods agreed upon in
    the Caltrack beta test. HDD and CDD are computed using fixed balance
    points, usage per day is computed for each month, and a data
    sufficiency requirement of at least 15 days of valid temperature+usage data
    is imposed on each month.
    '''
    def __init__(self, grid_search=False):
        if grid_search:
            self.bp_cdd = [50, 55, 60, 65, 70, 75, 80, 85]
            self.bp_hdd = [50, 55, 60, 65, 70, 75, 80, 85]
        else:
            self.bp_cdd, self.bp_hdd = [70, ], [60, ]

    def __repr__(self):
        return 'CaltrackFormatter()'

    def create_daily_data(self, data):
        ''' Helper function to handle monthly or other irregular data.'''
        if len(data.index) == 0:
            return pd.Series()
        idx = [data.index[0]]
        upd = []
        # Loop through the input data, skipping the first usage number,
        # and create a series of usage values by dividing equally across
        # each period.
        for i in data.index[1:]:
            start_date = idx[-1]
            ndays = (i - start_date).days
            this_upd = data.value[i] / float(ndays)
            for j in pd.date_range(start_date, i)[1:]:
                idx.append(j)
                upd.append(this_upd)
        idx = idx[1:]
        return pd.Series(upd, index=pd.DatetimeIndex(idx, freq='D'))

    def convert_to_monthly(self, df):
        # Convert from daily usage and temperature to monthly
        # usage per day and average HDD/CDD.
        cdd = {i: [0] for i in self.bp_cdd}
        hdd = {i: [0] for i in self.bp_hdd}
        if len(df.index) == 0:
            df_dict = {'upd': [], 'usage': [], 'ndays': []}
            df_dict.update({'CDD_' + str(bp): [] for bp in cdd.keys()})
            df_dict.update({'HDD_' + str(bp): [] for bp in hdd.keys()})
            return pd.DataFrame(df_dict, index=[])
        ndays, usage, upd, output_index = \
            [0], [0], [0], [df.index[0]]
        this_yr, this_mo = output_index[0].year, output_index[0].month
        for idx, row in df.iterrows():
            if this_yr != idx.year or this_mo != idx.month:
                ndays.append(0)
                usage.append(0)
                upd.append(0)
                for i in cdd.keys():
                    cdd[i].append(0)
                for i in hdd.keys():
                    hdd[i].append(0)
                this_yr, this_mo = idx.year, idx.month
                output_index.append(idx)
            if np.isfinite(row['energy']) and np.isfinite(row['tempF']):
                ndays[-1] = ndays[-1] + 1
                usage[-1] = usage[-1] + row['energy']
                for bp in cdd.keys():
                    cdd[bp][-1] = cdd[bp][-1] + \
                        np.maximum(row['tempF'] - bp, 0)
                for bp in hdd.keys():
                    hdd[bp][-1] = hdd[bp][-1] + \
                        np.maximum(bp - row['tempF'], 0)

        for i in range(len(usage)):
            if (ndays[i] < 15):
                # Caltrack sufficiency requirement of >=15 days per month
                upd[i] = np.nan
                for bp in cdd.keys():
                    cdd[bp][i] = np.nan
                for bp in hdd.keys():
                    hdd[bp][i] = np.nan
            else:
                upd[i] = (usage[i] / ndays[i])
                for bp in cdd.keys():
                    cdd[bp][i] = cdd[bp][i] / ndays[i]
                for bp in hdd.keys():
                    hdd[bp][i] = hdd[bp][i] / ndays[i]
        df_dict = {'upd': upd, 'usage': usage, 'ndays': ndays}
        df_dict.update({'CDD_' + str(bp): cdd[bp] for bp in cdd.keys()})
        df_dict.update({'HDD_' + str(bp): hdd[bp] for bp in hdd.keys()})
        output = pd.DataFrame(df_dict, index=output_index)
        return output

    def create_input(self, trace, weather_source):
        energy = pd.Series()
        if (trace.data.index.freq is None or
                to_offset(trace.data.index.freq) > to_offset('D')):
            # Input is less frequent than daily (e.g., monthly)
            energy = self.create_daily_data(trace.data)
        else:
            # Input is daily, hourly, 15-minutely, etc.
            energy = trace.data.value.resample('D').sum()

        tempF = weather_source.indexed_temperatures(energy.index, "degF")
        df = pd.DataFrame({"energy": energy, "tempF": tempF},
                          columns=["energy", "tempF"])
        df = self.convert_to_monthly(df)
        return df

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
        idx = None
        if (index.freq is None or
                to_offset(index.freq) > to_offset('D')):
            # Input is less frequent than daily (e.g., monthly)
            dummy = pd.DataFrame({'value': np.zeros(len(index))}, index=index)
            energy = self.create_daily_data(dummy)
            idx = energy.index
        else:
            # Input is daily, hourly, 15-minutely, etc.
            dummy = pd.Series(np.zeros(len(index)), index=index)
            energy = dummy.resample('D').sum()
            idx = energy.index

        tempF = weather_source.indexed_temperatures(idx, "degF")
        df = pd.DataFrame({"tempF": tempF, "energy": tempF * 0})
        df = self.convert_to_monthly(df)
        return df

    def serialize_input(self, input_data):
        return OrderedDict([
            (start.isoformat(), OrderedDict([
                (i, row[i] if np.isfinite(row[i]) else None)
                for i in row.keys()
            ]))
            for start, row in input_data.iterrows()
        ])

    def serialize_demand_fixture(self, demand_fixture_data):
        return OrderedDict([
            (start.isoformat(), OrderedDict([
                (i, row[i] if np.isfinite(row[i]) else None)
                for i in row.keys()
            ]))
            for start, row in demand_fixture_data.iterrows()
        ])

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
