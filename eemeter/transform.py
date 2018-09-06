from datetime import datetime, timedelta
from functools import partial

import numpy as np
import pandas as pd
import pytz

from .exceptions import NoBaselineDataError, NoReportingDataError
from .api import EEMeterWarning


__all__ = (
    "as_freq",
    "compute_temperature_features",
    "day_counts",
    "get_baseline_data",
    "get_reporting_data",
    "merge_temperature_data",
    "remove_duplicates",
    "overwrite_partial_rows_with_nan",
)


def _matching_groups(index, df, tolerance):
    # convert index to df for use with merge_asof
    index_df = pd.DataFrame({"index_col": index}, index=index)

    # get a dataframe containing mean temperature
    #   1) merge by matching temperature to closest previous meter start date,
    #      up to tolerance limit, using merge_asof.
    #   2) group by meter_index, and take the mean, ignoring all columns except
    #      the temperature column.
    groups = pd.merge_asof(
        left=df, right=index_df, left_index=True, right_index=True, tolerance=tolerance
    ).groupby("index_col")
    return groups


def _degree_day_columns(
    heating_balance_points,
    cooling_balance_points,
    degree_day_method,
    percent_hourly_coverage_per_day,
    percent_hourly_coverage_per_billing_period,
    use_mean_daily_values,
):
    # TODO(philngo): can this be refactored to be a more general without losing
    # on performance?

    # Not used in CalTRACK 2.0
    if degree_day_method == "hourly":

        def _compute_columns(temps):
            n_temps = temps.shape[0]
            n_temps_kept = temps.count()
            count_cols = {
                "n_hours_kept": n_temps_kept,
                "n_hours_dropped": n_temps - n_temps_kept,
            }
            if use_mean_daily_values:
                n_days = 1
            else:
                n_days = n_temps / 24.0
            cdd_cols = {
                "cdd_%s" % bp: np.maximum(temps - bp, 0).mean() * n_days
                for bp in cooling_balance_points
            }
            hdd_cols = {
                "hdd_%s" % bp: np.maximum(bp - temps, 0).mean() * n_days
                for bp in heating_balance_points
            }

            columns = count_cols
            columns.update(cdd_cols)
            columns.update(hdd_cols)
            return columns

    # CalTRACK 2.2.2.3
    n_limit_daily = 24 * percent_hourly_coverage_per_day

    if degree_day_method == "daily":

        def _compute_columns(temps):
            count = temps.shape[0]
            if count > 24:

                day_groups = np.floor(np.arange(count) / 24)
                daily_temps = temps.groupby(day_groups).agg(["mean", "count"])
                n_limit_period = percent_hourly_coverage_per_billing_period * count
                n_days_total = daily_temps.shape[0]

                # CalTrack 2.2.3.2
                if temps.notnull().sum() < n_limit_period:
                    daily_temps = daily_temps["mean"].iloc[0:0]
                else:
                    # CalTRACK 2.2.2.3
                    daily_temps = daily_temps["mean"][
                        daily_temps["count"] > n_limit_daily
                    ]
                n_days_kept = daily_temps.shape[0]
                count_cols = {
                    "n_days_kept": n_days_kept,
                    "n_days_dropped": n_days_total - n_days_kept,
                }

                if use_mean_daily_values:
                    n_days = 1
                else:
                    n_days = n_days_total

                cdd_cols = {
                    "cdd_%s" % bp: np.maximum(daily_temps - bp, 0).mean() * n_days
                    for bp in cooling_balance_points
                }
                hdd_cols = {
                    "hdd_%s" % bp: np.maximum(bp - daily_temps, 0).mean() * n_days
                    for bp in heating_balance_points
                }
            else:  # faster route for daily case, should have same effect.

                if count > n_limit_daily:
                    count_cols = {"n_days_kept": 1, "n_days_dropped": 0}
                    # CalTRACK 2.2.2.3
                    mean_temp = temps.mean()
                else:
                    count_cols = {"n_days_kept": 0, "n_days_dropped": 1}
                    mean_temp = np.nan

                # CalTrack 3.3.4.1.1
                cdd_cols = {
                    "cdd_%s" % bp: np.maximum(mean_temp - bp, 0)
                    for bp in cooling_balance_points
                }

                # CalTrack 3.3.5.1.1
                hdd_cols = {
                    "hdd_%s" % bp: np.maximum(bp - mean_temp, 0)
                    for bp in heating_balance_points
                }

            columns = count_cols
            columns.update(cdd_cols)
            columns.update(hdd_cols)
            return columns

    # TODO(philngo): option to ignore the count columns?

    agg_funcs = [("degree_day_columns", _compute_columns)]
    return agg_funcs


def merge_temperature_data(
    meter_data,
    temperature_data,
    heating_balance_points=None,
    cooling_balance_points=None,
    data_quality=False,
    temperature_mean=True,
    degree_day_method="daily",
    percent_hourly_coverage_per_day=0.5,
    percent_hourly_coverage_per_billing_period=0.9,
    use_mean_daily_values=True,
    tolerance=None,
    keep_partial_nan_rows=False,
):
    """ Merge meter data of any frequency with hourly temperature data to make
    a dataset to feed to models.

    Creates a :any:`pandas.DataFrame` with the same index as the meter data.

    .. note::

        For CalTRACK compliance (2.2.2.3), must set
        ``percent_hourly_coverage_per_day=0.5``,
        ``cooling_balance_points=range(30,90,X)``, and
        ``heating_balance_points=range(30,90,X)``, where
        X is either 1, 2, or 3. For natural gas meter use data, must
        set ``fit_cdd=False``.

    .. note::

        For CalTRACK compliance (2.2.3.2), for billing methods, must set
        ``percent_hourly_coverage_per_billing_period=0.9``.

    .. note::

        For CalTRACK compliance (2.3.3), ``meter_data`` and ``temperature_data``
        must both be timezone-aware and have matching timezones.

    .. note::

        For CalTRACK compliance (3.3.1.1), for billing methods, must set
        ``use_mean_daily_values=True``.

    .. note::

        For CalTRACK compliance (3.3.1.2), for daily or billing methods,
        must set ``degree_day_method=daily``.

    See also :any:`eemeter.compute_temperature_features`.

    Parameters
    ----------
    meter_data : :any:`pandas.DataFrame`
        DataFrame with :any:`pandas.DatetimeIndex` and a column with the name
        ``value``.
    temperature_data : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with hourly (``'H'``) frequency
        and a set of temperature values.
    cooling_balance_points : :any:`list` of :any:`int` or :any:`float`, optional
        List of cooling balance points for which to create cooling degree days.
    heating_balance_points : :any:`list` of :any:`int` or :any:`float`, optional
        List of heating balance points for which to create heating degree days.
    data_quality : :any:`bool`, optional
        If True, compute data quality columns for temperature, i.e.,
        ``temperature_not_null`` and ``temperature_null``, containing for
        each meter value
    temperature_mean : :any:`bool`, optional
        If True, compute temperature means for each meter period.
    degree_day_method : :any:`str`, ``'daily'`` or ``'hourly'``
        The method to use in calculating degree days.
    percent_hourly_coverage_per_day : :any:`str`, optional
        Percent hourly temperature coverage per day for heating and cooling
        degree days to not be dropped.
    use_mean_daily_values : :any:`bool`, optional
        If True, meter and degree day values should be mean daily values, not
        totals. If False, totals will be used instead.
    tolerance : :any:`pandas.Timedelta`, optional
        Do not merge more than this amount of temperature data beyond this limit.
    keep_partial_nan_rows: :any:`bool`, optional
        If True, keeps data in resultant :any:`pandas.DataFrame` that has
        missing temperature or meter data. Otherwise, these rows are overwritten
        entirely with ``numpy.nan`` values.

    Returns
    -------
    data : :any:`pandas.DataFrame`
        A dataset with the specified parameters.
    """
    # TODO(philngo): write fast route for hourly meter data + hourly temp data,
    #   possibly using pd.align or pd.reindex
    # TODO(philngo): think about providing some presets
    # TODO(ssuffian): fix the following: for billing period data when keep_partial_nan_rows=True, n_days_total is always one more than n_days_kept, due to the last row of the meter data being an np.nan value.

    freq_greater_than_daily = meter_data.index.freq is None or pd.Timedelta(
        meter_data.index.freq
    ) > pd.Timedelta("1D")

    meter_value_df = meter_data.value.to_frame("meter_value")

    # CalTrack 3.3.1.1
    # convert to average daily meter values.
    if use_mean_daily_values and freq_greater_than_daily:
        meter_value_df["meter_value"] = meter_value_df.meter_value / day_counts(
            meter_value_df.meter_value
        )

    temperature_feature_df = compute_temperature_features(
        temperature_data,
        meter_data.index,
        heating_balance_points=heating_balance_points,
        cooling_balance_points=cooling_balance_points,
        data_quality=data_quality,
        temperature_mean=temperature_mean,
        degree_day_method=degree_day_method,
        percent_hourly_coverage_per_day=percent_hourly_coverage_per_day,
        percent_hourly_coverage_per_billing_period=percent_hourly_coverage_per_billing_period,
        use_mean_daily_values=use_mean_daily_values,
        tolerance=tolerance,
        keep_partial_nan_rows=keep_partial_nan_rows,
    )

    df = pd.concat([meter_value_df, temperature_feature_df], axis=1)

    if not keep_partial_nan_rows:
        df = overwrite_partial_rows_with_nan(df)
    return df


def compute_temperature_features(
    temperature_data,
    meter_data_index,
    heating_balance_points=None,
    cooling_balance_points=None,
    data_quality=False,
    temperature_mean=True,
    degree_day_method="daily",
    percent_hourly_coverage_per_day=0.5,
    percent_hourly_coverage_per_billing_period=0.9,
    use_mean_daily_values=True,
    tolerance=None,
    keep_partial_nan_rows=False,
):
    """ Compute temperature features from hourly temperature data using the
    :any:`pandas.DatetimeIndex` meter data..

    Creates a :any:`pandas.DataFrame` with the same index as the meter data.

    .. note::

        For CalTRACK compliance (2.2.2.3), must set
        ``percent_hourly_coverage_per_day=0.5``,
        ``cooling_balance_points=range(30,90,X)``, and
        ``heating_balance_points=range(30,90,X)``, where
        X is either 1, 2, or 3. For natural gas meter use data, must
        set ``fit_cdd=False``.

    .. note::

        For CalTRACK compliance (2.2.3.2), for billing methods, must set
        ``percent_hourly_coverage_per_billing_period=0.9``.

    .. note::

        For CalTRACK compliance (2.3.3), ``meter_data`` and ``temperature_data``
        must both be timezone-aware and have matching timezones.

    .. note::

        For CalTRACK compliance (3.3.1.1), for billing methods, must set
        ``use_mean_daily_values=True``.

    .. note::

        For CalTRACK compliance (3.3.1.2), for daily or billing methods,
        must set ``degree_day_method=daily``.

    See also :any:`eemeter.merge_temperature_data`.

    Parameters
    ----------
    temperature_data : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with hourly (``'H'``) frequency
        and a set of temperature values.
    meter_data_index : :any:`pandas.DataFrame`
        A :any:`pandas.DatetimeIndex` corresponding to the index over which
        to compute temperature features.
    cooling_balance_points : :any:`list` of :any:`int` or :any:`float`, optional
        List of cooling balance points for which to create cooling degree days.
    heating_balance_points : :any:`list` of :any:`int` or :any:`float`, optional
        List of heating balance points for which to create heating degree days.
    data_quality : :any:`bool`, optional
        If True, compute data quality columns for temperature, i.e.,
        ``temperature_not_null`` and ``temperature_null``, containing for
        each meter value
    temperature_mean : :any:`bool`, optional
        If True, compute temperature means for each meter period.
    degree_day_method : :any:`str`, ``'daily'`` or ``'hourly'``
        The method to use in calculating degree days.
    percent_hourly_coverage_per_day : :any:`str`, optional
        Percent hourly temperature coverage per day for heating and cooling
        degree days to not be dropped.
    use_mean_daily_values : :any:`bool`, optional
        If True, meter and degree day values should be mean daily values, not
        totals. If False, totals will be used instead.
    tolerance : :any:`pandas.Timedelta`, optional
        Do not merge more than this amount of temperature data beyond this limit.
    keep_partial_nan_rows: :any:`bool`, optional
        If True, keeps data in resultant :any:`pandas.DataFrame` that has
        missing temperature or meter data. Otherwise, these rows are overwritten
        entirely with ``numpy.nan`` values.

    Returns
    -------
    data : :any:`pandas.DataFrame`
        A dataset with the specified parameters.
    """
    if temperature_data.index.freq != "H":
        raise ValueError(
            "temperature_data.index must have hourly frequency (freq='H')."
            " Found: {}".format(temperature_data.index.freq)
        )

    if not temperature_data.index.tz:
        raise ValueError(
            "temperature_data.index must be timezone-aware. You can set it with"
            " temperature_data.tz_localize(...)."
        )

    if meter_data_index.freq is None and meter_data_index.inferred_freq == "H":
        raise ValueError(
            "If you have hourly data explicitly set the frequency"
            " of the dataframe by setting"
            "``meter_data_index.freq ="
            " pd.tseries.frequencies.to_offset('H')."
        )

    if not meter_data_index.tz:
        raise ValueError(
            "meter_data_index must be timezone-aware. You can set it with"
            " meter_data.tz_localize(...)."
        )

    temp_agg_funcs = []
    temp_agg_column_renames = {}

    if heating_balance_points is None:
        heating_balance_points = []
    if cooling_balance_points is None:
        cooling_balance_points = []

    if tolerance is None and meter_data_index.freq is not None:
        tolerance = pd.Timedelta(meter_data_index.freq)

    if not (heating_balance_points == [] and cooling_balance_points == []):
        if degree_day_method == "hourly":
            pass
        elif degree_day_method == "daily":
            if meter_data_index.freq == "H":
                raise ValueError(
                    "degree_day_method='hourly' must be used with"
                    " hourly meter data. Found: 'daily'".format(degree_day_method)
                )
        else:
            raise ValueError("method not supported: {}".format(degree_day_method))

    # heating/cooling degree day aggregations. Needed for n_days fields as well.
    temp_agg_funcs.extend(
        _degree_day_columns(
            heating_balance_points=heating_balance_points,
            cooling_balance_points=cooling_balance_points,
            degree_day_method=degree_day_method,
            percent_hourly_coverage_per_day=percent_hourly_coverage_per_day,
            percent_hourly_coverage_per_billing_period=percent_hourly_coverage_per_billing_period,
            use_mean_daily_values=use_mean_daily_values,
        )
    )
    temp_agg_column_renames.update(
        {("temp", "degree_day_columns"): "degree_day_columns"}
    )

    if data_quality:
        temp_agg_funcs.extend(
            [("not_null", "count"), ("null", lambda x: x.isnull().sum())]
        )
        temp_agg_column_renames.update(
            {
                ("temp", "not_null"): "temperature_not_null",
                ("temp", "null"): "temperature_null",
            }
        )

    if temperature_mean:
        temp_agg_funcs.extend([("mean", "mean")])
        temp_agg_column_renames.update({("temp", "mean"): "temperature_mean"})

    # aggregate temperatures
    temp_df = temperature_data.to_frame("temp")
    temp_groups = _matching_groups(meter_data_index, temp_df, tolerance)
    temp_aggregations = temp_groups.agg({"temp": temp_agg_funcs})

    # expand temp aggregations by faking and delete the `meter_value` column.
    # I haven't yet figured out a way to avoid this and get the desired
    # structure and behavior. (philngo)
    meter_value = pd.DataFrame({"meter_value": 0}, index=meter_data_index)
    df = pd.concat([meter_value, temp_aggregations], axis=1).rename(
        columns=temp_agg_column_renames
    )
    del df["meter_value"]

    if df.empty:
        if "degree_day_columns" in df:
            column_defaults = {
                column: [] for column in ["n_days_dropped", "n_days_kept"]
            }
            df = df.drop(["degree_day_columns"], axis=1).assign(**column_defaults)
    else:
        # expand degree_day_columns
        if "degree_day_columns" in df:
            df = pd.concat(
                [
                    df.drop(["degree_day_columns"], axis=1),
                    df["degree_day_columns"].dropna().apply(pd.Series),
                ],
                axis=1,
            )

    if not keep_partial_nan_rows:
        df = overwrite_partial_rows_with_nan(df)

    return df


def overwrite_partial_rows_with_nan(df):
    return df.dropna().reindex(df.index)


def remove_duplicates(df_or_series):
    """ Remove duplicate rows or values by keeping the first of each duplicate.

    Parameters
    ----------
    df_or_series : :any:`pandas.DataFrame` or :any:`pandas.Series`
        Pandas object from which to drop duplicate index values.

    Returns
    -------
    deduplicated : :any:`pandas.DataFrame` or :any:`pandas.Series`
        The deduplicated pandas object.
    """
    # CalTrack 2.3.2.2
    return df_or_series[~df_or_series.index.duplicated(keep="first")]


def as_freq(meter_data_series, freq, atomic_freq="1 Min"):
    """Resample meter data to a different frequency.

    This method can be used to upsample or downsample meter data. The
    assumption it makes to do so is that meter data is constant and averaged
    over the given periods. For instance, to convert billing-period data to
    daily data, this method first upsamples to the atomic frequency
    (1 minute freqency, by default), "spreading" usage evenly across all
    minutes in each period. Then it downsamples to hourly frequency and
    returns that result.

    **Caveats**:

     - This method gives a fair amount of flexibility in
       resampling as long as you are OK with the assumption that usage is
       constant over the period (this assumption is generally broken in
       observed data at large enough frequencies, so this caveat should not be
       taken lightly).

     - This method should not be used for sampled (e.g., temperature data)
       rather than recorded data (e.g., meter data), as sampled data cannot be
       "spread" in the same way.

    Parameters
    ----------
    meter_data_series : :any:`pandas.Series`
        Meter data to resample. Should have a :any:`pandas.DatetimeIndex`.
    freq : :any:`str`
        The frequency to resample to. This should be given in a form recognized
        by the :any:`pandas.Series.resample` method.
    atomic_freq : :any:`str`, optional
        The "atomic" frequency of the intermediate data form. This can be
        adjusted to a higher atomic frequency to increase speed or memory
        performance.

    Returns
    -------
    resampled_meter_data : :any:`pandas.Series`
        Meter data resampled to the given frequency.
    """
    # TODO(philngo): make sure this complies with CalTRACK 2.2.2.1
    if not isinstance(meter_data_series, pd.Series):
        raise ValueError(
            "expected series, got object with class {}".format(
                meter_data_series.__class__
            )
        )
    if meter_data_series.empty:
        return meter_data_series
    series = remove_duplicates(meter_data_series)
    target_freq = pd.Timedelta(atomic_freq)
    timedeltas = (series.index[1:] - series.index[:-1]).append(
        pd.TimedeltaIndex([pd.NaT])
    )
    spread_factor = target_freq.total_seconds() / timedeltas.total_seconds()
    series_spread = series * spread_factor
    atomic_series = series_spread.asfreq(atomic_freq, method="ffill")
    resampled = atomic_series.resample(freq).sum()
    return resampled


def day_counts(series):
    """Days between index datetime values as a :any:`pandas.Series`.

    Parameters
    ----------
    series : :any:`pandas.Series` with :any:`pandas.DatetimeIndex`
        A series for which to get day counts.
    Returns
    -------
    day_counts : :any:`pandas.Series`
        A :any:`pandas.Series` with counts of days between periods. Counts are
        given on start dates of periods.
    """
    # TODO(philngo): incorporate this directly into merge_temperature_data

    # dont affect the original data
    series = series.copy()

    if len(series) == 0:
        return pd.Series([], index=series.index)

    timedeltas = (series.index[1:] - series.index[:-1]).append(
        pd.TimedeltaIndex([pd.NaT])
    )
    timedelta_days = timedeltas.total_seconds() / (60 * 60 * 24)

    return pd.Series(timedelta_days, index=series.index)


def get_baseline_data(data, start=None, end=None, max_days=365):
    """ Filter down to baseline period data.

    .. note::

        For compliance with CalTRACK, set ``max_days=365`` (section 2.2.1.1).

    Parameters
    ----------
    data : :any:`pandas.DataFrame` or :any:`pandas.Series`
        The data to filter to baseline data. This data will be filtered down
        to an acceptable baseline period according to the dates passed as
        `start` and `end`, or the maximum period specified with `max_days`.
    start : :any:`datetime.datetime`
        A timezone-aware datetime that represents the earliest allowable start
        date for the baseline data. The stricter of this or `max_days` is used
        to determine the earliest allowable baseline period date.
    end : :any:`datetime.datetime`
        A timezone-aware datetime that represents the latest allowable end
        date for the baseline data, i.e., the latest date for which data is
        available before the intervention begins.
    max_days : :any:`int`
        The maximum length of the period. Ignored if `end` is not set.
        The stricter of this or `start` is used to determine the earliest
        allowable baseline period date.

    Returns
    -------
    baseline_data, warnings : :any:`tuple` of (:any:`pandas.DataFrame` or :any:`pandas.Series`, :any:`list` of :any:`eemeter.EEMeterWarning`)
        Data for only the specified baseline period and any associated warnings.
    """

    start_inf = False
    if start is None:
        # py datetime min/max are out of range of pd.Timestamp min/max
        start = pytz.UTC.localize(pd.Timestamp.min)
        start_inf = True

    end_inf = False
    if end is None:
        end = pytz.UTC.localize(pd.Timestamp.max)
        end_inf = True
    else:
        if max_days is not None:
            min_start = end - timedelta(days=max_days)
            if start < min_start:
                start = min_start

    warnings = []
    # warn if there is a gap at end
    data_end = data.index.max()
    if not end_inf and data_end < end:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_baseline_data.gap_at_baseline_end",
                description=(
                    "Data does not have coverage at requested baseline end date."
                ),
                data={
                    "requested_end": end.isoformat(),
                    "data_end": data_end.isoformat(),
                },
            )
        )

    # warn if there is a gap at start
    data_start = data.index.min()
    if not start_inf and start < data_start:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_baseline_data.gap_at_baseline_start",
                description=(
                    "Data does not have coverage at requested baseline start date."
                ),
                data={
                    "requested_start": start.isoformat(),
                    "data_start": data_start.isoformat(),
                },
            )
        )

    # copying prevents setting on slice warnings
    baseline_data = data[start:end].copy()

    if baseline_data.empty:
        raise NoBaselineDataError()

    baseline_data.iloc[-1] = np.nan

    return baseline_data, warnings


def get_reporting_data(data, start=None, end=None, max_days=365):
    """ Filter down to reporting period data.

    Parameters
    ----------
    data : :any:`pandas.DataFrame` or :any:`pandas.Series`
        The data to filter to reporting data. This data will be filtered down
        to an acceptable reporting period according to the dates passed as
        `start` and `end`, or the maximum period specified with `max_days`.
    start : datetime.datetime
        A timezone-aware datetime that represents the earliest allowable start
        date for the reporting data, i.e., the earliest date for which data is
        available after the intervention begins.
    end : datetime.datetime
        A timezone-aware datetime that represents the latest allowable end
        date for the reporting data. The stricter of this or `max_days` is used
        to determine the latest allowable reporting period date.
    max_days : int
        The maximum length of the period. Ignored if `start` is not set.
        The stricter of this or `end` is used to determine the latest
        allowable reporting period date.

    Returns
    -------
    reporting_data, warnings : :any:`tuple` of (:any:`pandas.DataFrame` or :any:`pandas.Series`, :any:`list` of :any:`eemeter.EEMeterWarning`)
        Data for only the specified reporting period and any associated warnings.
    """
    # TODO(philngo): use default max_days None? Maybe too symmetrical with
    # get_baseline_data?

    end_inf = False
    if end is None:
        # py datetime min/max are out of range of pd.Timestamp min/max
        end = pytz.UTC.localize(pd.Timestamp.max)
        end_inf = True

    start_inf = False
    if start is None:
        start = pytz.UTC.localize(pd.Timestamp.min)
        start_inf = True
    else:
        if max_days is not None:
            max_end = start + timedelta(days=max_days)
            if end > max_end:
                end = max_end

    warnings = []
    # warn if there is a gap at end
    data_end = data.index.max()
    if not end_inf and data_end < end:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_reporting_data.gap_at_reporting_end",
                description=(
                    "Data does not have coverage at requested reporting end date."
                ),
                data={
                    "requested_end": end.isoformat(),
                    "data_end": data_end.isoformat(),
                },
            )
        )

    # warn if there is a gap at start
    data_start = data.index.min()
    if not start_inf and start < data_start:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_reporting_data.gap_at_reporting_start",
                description=(
                    "Data does not have coverage at requested reporting start date."
                ),
                data={
                    "requested_start": start.isoformat(),
                    "data_start": data_start.isoformat(),
                },
            )
        )

    # copying prevents setting on slice warnings
    reporting_data = data[start:end].copy()

    if reporting_data.empty:
        raise NoReportingDataError()

    reporting_data.iloc[-1] = np.nan

    return reporting_data, warnings
