import numpy as np
import pandas as pd

__all__ = (
    "meter_data_from_csv",
    "meter_data_from_json",
    "meter_data_to_csv",
    "temperature_data_from_csv",
    "temperature_data_from_json",
    "temperature_data_to_csv",
)


def meter_data_from_csv(
    filepath_or_buffer,
    tz=None,
    start_col="start",
    value_col="value",
    gzipped=False,
    freq=None,
    **kwargs
):
    """ Load meter data from a CSV file.

    Default format::

        start,value
        2017-01-01T00:00:00+00:00,0.31
        2017-01-02T00:00:00+00:00,0.4
        2017-01-03T00:00:00+00:00,0.58

    Parameters
    ----------
    filepath_or_buffer : :any:`str` or file-handle
        File path or object.
    tz : :any:`str`, optional
        E.g., ``'UTC'`` or ``'US/Pacific'``
    start_col : :any:`str`, optional, default ``'start'``
        Date period start column.
    value_col : :any:`str`, optional, default ``'value'``
        Value column, can be in any unit.
    gzipped : :any:`bool`, optional
        Whether file is gzipped.
    freq : :any:`str`, optional
        If given, apply frequency to data using :any:`pandas.DataFrame.resample`.
    **kwargs
        Extra keyword arguments to pass to :any:`pandas.read_csv`, such as
        ``sep='|'``.
    """

    read_csv_kwargs = {
        "usecols": [start_col, value_col],
        "dtype": {value_col: np.float64},
        "parse_dates": [start_col],
        "index_col": start_col,
    }

    if gzipped:
        read_csv_kwargs.update({"compression": "gzip"})

    # allow passing extra kwargs
    read_csv_kwargs.update(kwargs)

    df = pd.read_csv(filepath_or_buffer, **read_csv_kwargs).tz_localize("UTC")
    if tz is not None:
        df = df.tz_convert(tz)

    if freq == "hourly":
        df = df.resample("H").sum()
    elif freq == "daily":
        df = df.resample("D").sum()

    return df


def temperature_data_from_csv(
    filepath_or_buffer,
    tz=None,
    date_col="dt",
    temp_col="tempF",
    gzipped=False,
    freq=None,
    **kwargs
):
    """ Load temperature data from a CSV file.

    Default format::

        dt,tempF
        2017-01-01T00:00:00+00:00,21
        2017-01-01T01:00:00+00:00,22.5
        2017-01-01T02:00:00+00:00,23.5

    Parameters
    ----------
    filepath_or_buffer : :any:`str` or file-handle
        File path or object.
    tz : :any:`str`, optional
        E.g., ``'UTC'`` or ``'US/Pacific'``
    date_col : :any:`str`, optional, default ``'dt'``
        Date period start column.
    temp_col : :any:`str`, optional, default ``'tempF'``
        Temperature column.
    gzipped : :any:`bool`, optional
        Whether file is gzipped.
    freq : :any:`str`, optional
        If given, apply frequency to data using :any:`pandas.Series.resample`.
    **kwargs
        Extra keyword arguments to pass to :any:`pandas.read_csv`, such as
        ``sep='|'``.
    """
    read_csv_kwargs = {
        "usecols": [date_col, temp_col],
        "dtype": {temp_col: np.float64},
        "parse_dates": [date_col],
        "index_col": date_col,
    }

    if gzipped:
        read_csv_kwargs.update({"compression": "gzip"})

    # allow passing extra kwargs
    read_csv_kwargs.update(kwargs)

    if tz is None:
        tz = "UTC"

    df = pd.read_csv(filepath_or_buffer, **read_csv_kwargs).tz_localize(tz)

    if freq == "hourly":
        df = df.resample("H").sum()

    return df[temp_col]


def meter_data_from_json(data, orient="list"):
    """ Load meter data from json.

    Default format::

        [
            ['2017-01-01T00:00:00+00:00', 3.5],
            ['2017-02-01T00:00:00+00:00', 0.4],
            ['2017-03-01T00:00:00+00:00', 0.46],
        ]

    Parameters
    ----------
    data : :any:`list`
        List elements are each a rows of data.

    Returns
    -------
    df : :any:`pandas.DataFrame`
        DataFrame with a single column (``'value'``) and a
        :any:`pandas.DatetimeIndex`.
    """
    if orient == "list":
        df = pd.DataFrame(data, columns=["start", "value"])
        df["start"] = pd.DatetimeIndex(df.start).tz_localize("UTC")
        df = df.set_index("start")
        return df
    else:
        raise ValueError("orientation not recognized.")


def temperature_data_from_json(data, orient="list"):
    """ Load temperature data from json. (Must be given in degrees
    Fahrenheit).

    Default format::

        [
            ['2017-01-01T00:00:00+00:00', 3.5],
            ['2017-01-01T01:00:00+00:00', 5.4],
            ['2017-01-01T02:00:00+00:00', 7.4],
        ]

    Parameters
    ----------
    data : :any:`list`
        List elements are each a rows of data.

    Returns
    -------
    series : :any:`pandas.Series`
        DataFrame with a single column (``'tempF'``) and a
        :any:`pandas.DatetimeIndex`.
    """
    if orient == "list":
        df = pd.DataFrame(data, columns=["dt", "tempF"])
        series = df.tempF
        series.index = pd.DatetimeIndex(df.dt).tz_localize("UTC")
        return series
    else:
        raise ValueError("orientation not recognized.")


def meter_data_to_csv(meter_data, path_or_buf):
    """ Write meter data to CSV. See also :any:`pandas.DataFrame.to_csv`.

    Parameters
    ----------
    meter_data : :any:`pandas.DataFrame`
        Meter data DataFrame with ``'value'`` column and
        :any:`pandas.DatetimeIndex`.
    path_or_buf : :any:`str` or file handle, default None
        File path or object, if None is provided the result is returned as a string.
    """
    if meter_data.index.name is None:
        meter_data.index.name = "start"
    return meter_data.to_csv(path_or_buf, index=True)


def temperature_data_to_csv(temperature_data, path_or_buf):
    """ Write temperature data to CSV. See also :any:`pandas.DataFrame.to_csv`.

    Parameters
    ----------
    temperature_data : :any:`pandas.Series`
        Temperature data series with :any:`pandas.DatetimeIndex`.
    path_or_buf : :any:`str` or file handle, default None
        File path or object, if None is provided the result is returned as a string.
    """
    if temperature_data.index.name is None:
        temperature_data.index.name = "dt"
    if temperature_data.name is None:
        temperature_data.name = "temperature"
    return temperature_data.to_frame().to_csv(path_or_buf, index=True)
