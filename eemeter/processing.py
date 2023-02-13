#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2023 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""


import pandas as pd
from dateutil.relativedelta import relativedelta
import eemeter

__all__ = (
    "add_freq",
    "trim",
    "sum_gas_and_elec",
    "format_energy_data_for_eemeter",
    "format_temperature_data_for_eemeter",
    "eemeter_hourly",
    "eemeter_daily",
)


def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

     Returns a copy.  If `freq` is None, it is inferred.

     Note: this function is taken from
     https://stackoverflow.com/questions/46217529/pandas-datetimeindex-frequency-is-none-and-cant-be-set;
     credit Brad Solomon.


    Parameters
    ----------
    idx : :any:`pandas.DateTimeIndex`
        Any DateTimeIndex.
    freq : :any valid DateTimeIndex Freq in 'str' format
        The frequency of the datetime index. Defaults to 'None' if frequency is to be inferred.

    Returns
    -------
    idx : :any:`pandas.DateTimeIndex`
        A copy of idx with frequency added.
    """

    idx = idx.copy()
    if freq is None:
        if idx.freq is None:
            freq = pd.infer_freq(idx)
        else:
            return idx
    idx.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.freq is None:
        raise AttributeError(
            "no discernible frequency found to `idx`.  Specify"
            " a frequency string with `freq`."
        )
    return idx


def trim(*args, freq="H", tz="UTC"):
    """A helper function which trims a given number of time series dataframes so that they all correspond to the same
    time periods. Typically used to ensure that both gas, electricity, and temperature datasets cover the same time
    period. Trim undertakes the following steps:

       - copies dataframes
       - sets indexes to datetimes if not already
       - localises index to UTC (default - if other timezone applies this should be specified)
       - sorts in ascending order against DateTimeIndex
       - drops nulls at both start and end of df
       - trims the dataframes by equalising all min(df.index) and max(df.index)

       Trim requires both input dataframes to have some degree of overlap beforehand.

     Parameters
    ----------
    *args : : one or more 'pandas.DataFrame's
        A set of regular time series datasets. If index is not DateTimeIndex, function will convert accordingly. There
        must be overlap between all datasets otherwise trim will return IndexError. Can function with one dataframe,
        though not much point given functionality.
    freq : : any valid DateTimeIndex frequency.
        This is used to identify any duplicates and missing values in a dataframe and ensure that each dataframe in the
        returned tuple is of the same length. Freq defaults to '1H' (one hour) but can be, for example '0.5H' (1/2 hour)
    tz : : any valid timezone 'str'
        The timezone associated with the given dataframes. If timezone-naive, function will localise to 'UTC' as
        default.

     Returns
    -------
     out_dfs : :any:`tuple` of 'pandas.DataFrame's.
         A list of dataframes trimmed to equal total intervals, arranged in eemeter format (i.e. with ascending
         indices).
    """
    new_tuple = ()
    if len(list(args)) == 1:
        args = args[0]
    for i in args:
        df = i
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, infer_datetime_format=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz=tz)  # defaults to UTC
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        df = df.resample(freq).asfreq()
        new_tuple = new_tuple + (df,)
    max_start = max([min(df.index) for df in new_tuple])
    min_end = min([max(df.index) for df in new_tuple])
    out_dfs = ()
    if max_start < min_end:
        for df in new_tuple:
            out_dfs = out_dfs + (df.loc[max_start:min_end],)
    else:
        raise IndexError("Trim requires for all dfs to have some overlap.")

    return out_dfs


def sum_gas_and_elec(gas, elec):
    """A helper function which sums kWh gas and electricity data to account for whole-building analysis using eemeter.

    Parameters
    ----------
    gas : :any:`pandas.DataFrame`
        Gas time series data, unit kWh.
    elec : :any:`pandas.DataFrame`
         Electricity time series data, unit kWh.

    Returns
    -------
    total : :any:`pandas.DataFrame`
        Total gas and electricity consumption time series data.
    """

    if gas is None:
        if not isinstance(elec.iloc[:, 0], float):
            elec.iloc[:, 0] = pd.to_numeric(elec.iloc[:, 0])
        total = elec
        return total
    if elec is None:
        if not isinstance(gas.iloc[0, 0], float):
            gas.iloc[:, 0] = pd.to_numeric(gas.iloc[:, 0])
        total = gas
        return total
    else:
        if not isinstance(gas.iloc[0, 0], float):
            gas.iloc[:, 0] = pd.to_numeric(gas.iloc[:, 0])
        if not isinstance(elec.iloc[:, 0], float):
            elec.iloc[:, 0] = pd.to_numeric(elec.iloc[:, 0])
        total = gas.join(elec.iloc[:, 0], rsuffix="_elec", lsuffix="_gas")
        total["value"] = total.sum(axis=1)
        total = total.iloc[:, -1:]
        return total


def _check_input_formatting(input, tz="UTC"):
    if not isinstance(input.index, pd.DatetimeIndex):
        if isinstance(input.index, pd.RangeIndex):
            for i in [
                "start",
                "Start",
                "Datetime",
                "timestamp",
                "Timestamp",
                "datetime",
            ]:  # this is a non-exhaustive list (welcome additions) of possible timestamp headers when not in index.
                if i in input.columns.values:
                    input = input.set_index(i)
        if not isinstance(input.index, pd.DatetimeIndex):
            input.index = pd.to_datetime(input.index)
            if input.index[0].tzinfo is None:
                input.index = input.index.tz_localize(tz=tz)
        else:
            raise ValueError(
                "Data is not in correct format - index should be of class 'pd.core.indexes.datetimes.DatetimeIndex',"
                + " or datetime column should be labelled one of: 'Start', 'start', 'Datetime', 'timestamp', "
                  "'Timestamp', or 'Datetime'."
            )
    if input.index[0].tzinfo is None:
        input.index = input.index.tz_localize(tz=tz)
    return input


def _format_data_for_eemeter_hourly(df, tz="UTC"):
    if df is not None:
        df = df.copy()
        df = _check_input_formatting(df, tz)
        df = df.sort_index()
        return df
    else:
        return None


def format_energy_data_for_eemeter(*args, method="hourly", tz="UTC"):
    """A helper function which ensures energy consumption data is formatted for eemeter processing.

    Parameters
    ----------
    *args : :one or more `pandas.DataFrame`s
        Energy consumption time series data. Consumption must be measured in the same units.
    method : : any valid 'str'
        The relevant eemeter model requiring formatting. Must be either 'hourly', 'daily', or 'billing'. Defaults to
        'hourly'.
    tz : : any valid timezone 'str'
        The timezone associated with the given dataframes. If timezone-naive, function will localise to 'UTC' as
        default.

    Returns
    -------
    args_tuple : any 'list' containing one or more 'pandas.DataFrame's
        A list of dataframes comprising energy consumption data in eemeter format.
    """

    if method == "hourly":
        freq = "H"
    elif method == "daily":
        freq = "D"
    elif method == "billing":
        freq = "M"
    else:
        raise ValueError("'method' must be either 'hourly', 'daily' or 'billing'.")

    args_tuple = ()
    for df in args:
        df = _format_data_for_eemeter_hourly(df, tz)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df = df.resample(freq).sum()
        df.index = df.index.rename("start")
        args_tuple = args_tuple + (df,)
        if df.columns[0] != "value":
            current_col_name = df.columns[0]
            df.rename(columns={current_col_name: "value"}, inplace=True)

    if len(args_tuple) == 1:
        return args_tuple[0]
    else:
        args_list = list(args_tuple)
        args_list[-1], args_list[-2] = trim(args_list[-1], args_list[-2], freq=freq)
        args_tuple = tuple(args_list)
        return args_tuple


def format_temperature_data_for_eemeter(temperature_data, tz="UTC"):
    """A helper function which ensures external temperature data is formatted for eemeter processing.

    Parameters
    ----------
    temperature_data : :any:``
        Hourly external temperature data. If DataFrame, not pd.Series (as required by CalTRACK) function will convert.
    tz : : any valid timezone 'str'
        The timezone associated with the given dataframes. If timezone-naive, function will localise to 'UTC' as
        default.
    Returns
    -------
    temperature_data : :any:``
        Hourly external temperature data in eemeter format.
    """

    temperature_data = _format_data_for_eemeter_hourly(temperature_data, tz)
    mask = temperature_data.index.minute == 00
    temperature_data = temperature_data[mask]
    if temperature_data.index.freq == None:
        temperature_data.index = add_freq(temperature_data.index)
    if isinstance(temperature_data, pd.DataFrame):
        temperature_data = temperature_data.squeeze()
    return temperature_data


def eemeter_hourly(
    gas,
    elec,
    temperature_data,
    blackout_start_date,
    blackout_end_date,
    region: str = "USA",
):
    """An output function which takes gas, electricity, external temperature data, blackout start and end dates, and
    returns a metered savings dataframe for the period between the blackout end date and today.

    Parameters
    ----------
    gas : :any:`pandas.DataFrame`
        Gas time series data, unit kWh.
    elec : :any:`pandas.DataFrame`
        Electricity time series data, unit kWh.
    temperature_data : :any:``
        Hourly external temperature data. If DataFrame, not pd.Series (as required by CalTRACK) function will convert.
    blackout_start_date : :any: 'datetime.datetime'
        The date at which improvement works commenced.
    blackout_end_date : :any: 'datetime.datetime'
        The date by which improvement works completed and metering resumed.
    region : :any 'str'
        The relevant region of the world. See eemeter/region_info.csv for options.
        Defaults to 'USA' unless otherwise specified for alignment with eeweather
        conventions.

    Returns
    -------
    metered_savings_dataframe: :any:`pandas.DataFrame`
    DataFrame with metered savings, indexed with
    ``reporting_meter_data.index``. Will include the following columns:

     - ``counterfactual_usage`` (baseline model projected into reporting period)
     - ``reporting_observed`` (given by reporting_meter_data)
     - ``metered_savings``

     If `with_disaggregated` is set to True, the following columns will also
     be in the results DataFrame:

     - ``counterfactual_base_load``
     - ``counterfactual_heating_load``
     - ``counterfactual_cooling_load``
    """
    meter_data = sum_gas_and_elec(gas, elec)

    baseline_meter_data, baseline_warnings = eemeter.get_baseline_data(
        meter_data,
        start=blackout_start_date - relativedelta(years=1),
        end=blackout_end_date,
        max_days=None,
    )

    # create a design matrix for occupancy and segmentation
    preliminary_design_matrix = eemeter.create_caltrack_hourly_preliminary_design_matrix(
        baseline_meter_data, temperature_data, region
    )

    # build 12 monthly models - each step from now on operates on each segment
    segmentation = eemeter.segment_time_series(
        preliminary_design_matrix.index, "three_month_weighted"
    )

    # assign an occupancy status to each hour of the week (0-167)
    occupancy_lookup = eemeter.estimate_hour_of_week_occupancy(
        preliminary_design_matrix, segmentation=segmentation, region=region
    )

    # assign temperatures to bins
    (
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    ) = eemeter.fit_temperature_bins(
        preliminary_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )

    # build a design matrix for each monthly segment
    segmented_design_matrices = eemeter.create_caltrack_hourly_segmented_design_matrices(
        preliminary_design_matrix,
        segmentation,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )

    # build a CalTRACK hourly model
    baseline_model = eemeter.fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    # get a year of reporting period data
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )

    # compute metered savings for the year of the reporting period we've selected
    metered_savings_dataframe, error_bands = eemeter.metered_savings(
        baseline_model,
        reporting_meter_data,
        temperature_data,
        with_disaggregated=True,
        region=region,
    )

    return metered_savings_dataframe


def eemeter_daily(
    gas,
    elec,
    temperature_data,
    blackout_start_date,
    blackout_end_date,
    region: str = "USA",
):

    meter_data = sum_gas_and_elec(gas, elec)

    # get meter data suitable for fitting a baseline model
    baseline_meter_data, warnings = eemeter.get_baseline_data(
        meter_data,
        start=blackout_start_date - relativedelta(years=1),
        end=blackout_end_date,
        max_days=None,
    )

    # create a design matrix (the input to the model fitting step)
    baseline_design_matrix = eemeter.create_caltrack_daily_design_matrix(
        baseline_meter_data, temperature_data, region=region
    )

    # build a CalTRACK model
    baseline_model = eemeter.fit_caltrack_usage_per_day_model(baseline_design_matrix)

    # get a year of reporting period data
    reporting_meter_data, warnings = eemeter.get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )

    # compute metered savings for the year of the reporting period we've selected
    metered_savings_dataframe, error_bands = eemeter.metered_savings(
        baseline_model,
        reporting_meter_data,
        temperature_data,
        with_disaggregated=True,
        region=region,
    )

    return metered_savings_dataframe
