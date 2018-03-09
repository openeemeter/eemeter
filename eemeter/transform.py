from datetime import datetime, timedelta
from functools import partial

import numpy as np
import pandas as pd
import pytz

from .exceptions import NoBaselineDataError, NoReportingDataError


__all__ = (
    'billing_as_daily',
    'get_baseline_data',
    'get_reporting_data',
    'merge_temperature_data',
    'day_counts',
)


def _matching_groups(index, df):
    # convert index to df for use with merge_asof
    index_df = pd.DataFrame({'index_col': index}, index=index)

    # get a dataframe containing mean temperature
    #   1) merge by matching temperature to closest previous meter start date,
    #      up to tolerance limit, using merge_asof.
    #   2) group by meter_index, and take the mean, ignoring all columns except
    #      the temperature column.
    groups = pd.merge_asof(
        left=df, right=index_df,
        left_index=True, right_index=True,
    ).groupby('index_col')
    return groups


def _degree_day_columns(
    heating_balance_points, cooling_balance_points, degree_day_method,
    percent_hourly_coverage_per_day, use_mean_daily_values,
):
    # TODO(philngo): can this be refactored to be a more general without losing
    # on performance?

    if degree_day_method == 'hourly':
        def _compute_columns(temps):
            n_temps = temps.shape[0]
            n_temps_kept = temps.count()
            count_cols = {
                'n_hours_kept': n_temps_kept,
                'n_hours_dropped': n_temps - n_temps_kept,
            }
            if use_mean_daily_values:
                n_days = 1
            else:
                n_days = n_temps / 24.0
            cdd_cols = {
                'cdd_%s' % bp: np.maximum(temps - bp, 0).mean() * n_days
                for bp in cooling_balance_points
            }
            hdd_cols = {
                'hdd_%s' % bp: np.maximum(bp - temps, 0).mean() * n_days
                for bp in heating_balance_points
            }

            columns = count_cols
            columns.update(cdd_cols)
            columns.update(hdd_cols)
            return columns

    n_limit = 24 * percent_hourly_coverage_per_day

    if degree_day_method == 'daily':
        def _compute_columns(temps):
            count = temps.shape[0]
            if count > 24:

                day_groups = np.floor(np.arange(count) / 24)
                daily_temps = temps.groupby(day_groups).agg(['mean', 'count'])
                n_days_total = daily_temps.shape[0]
                daily_temps = daily_temps['mean'][daily_temps['count'] > n_limit]
                n_days_kept = daily_temps.shape[0]
                count_cols = {
                    'n_days_kept': n_days_kept,
                    'n_days_dropped': n_days_total - n_days_kept,
                }

                if use_mean_daily_values:
                    n_days = 1
                else:
                    n_days = n_days_total

                cdd_cols = {
                    'cdd_%s' % bp: np.maximum(daily_temps - bp, 0).mean() * n_days
                    for bp in cooling_balance_points
                }
                hdd_cols = {
                    'hdd_%s' % bp: np.maximum(bp - daily_temps, 0).mean() * n_days
                    for bp in heating_balance_points
                }
            else:  # faster route for daily case, should have same effect.

                if count > n_limit:
                    count_cols = {
                        'n_days_kept': 1,
                        'n_days_dropped': 0,
                    }
                    mean_temp = temps.mean()
                else:
                    count_cols = {
                        'n_days_kept': 0,
                        'n_days_dropped': 1,
                    }
                    mean_temp = np.nan

                cdd_cols = {
                    'cdd_%s' % bp: np.maximum(mean_temp - bp, 0)
                    for bp in cooling_balance_points
                }
                hdd_cols = {
                    'hdd_%s' % bp: np.maximum(bp - mean_temp, 0)
                    for bp in heating_balance_points
                }

            columns = count_cols
            columns.update(cdd_cols)
            columns.update(hdd_cols)
            return columns

    agg_funcs = [('degree_day_columns', _compute_columns)]
    return agg_funcs


def merge_temperature_data(
    meter_data, temperature_data, heating_balance_points=None,
    cooling_balance_points=None, data_quality=False, temperature_mean=True,
    degree_day_method='daily', percent_hourly_coverage_per_day=0.5,
    use_mean_daily_values=True,
):
    ''' Merge meter data of any frequency with hourly temperature data to make
    a dataset to feed to models.

    Creates a :any:`pandas.DataFrame` with the same index as the meter data.

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
    use_mean_daily_values : :any:`bool`
        If True, meter and degree day values should be mean daily values, not
        totals. If False, totals will be used instead.

    Returns
    -------
    data : :any:`pandas.DataFrame`
        A dataset with the specified parameters.
    '''
    # TODO(philngo): write fast route for hourly meter data + hourly temp data

    temp_agg_funcs = []
    temp_agg_column_renames = {}

    if heating_balance_points is None:
        heating_balance_points = []
    if cooling_balance_points is None:
        cooling_balance_points = []

    if not (heating_balance_points == [] and cooling_balance_points == []):

        if degree_day_method == 'hourly':
            pass
        elif degree_day_method == 'daily':
            if meter_data.index.freq == 'H':
                raise ValueError(
                    "degree_day_method='hourly' must be used with"
                    " hourly meter data. Found: 'daily'".format(degree_day_method)
                )
        else:
            raise ValueError('method not supported: {}'.format(degree_day_method))

        # heating/cooling degree day aggregations
        temp_agg_funcs.extend(_degree_day_columns(
            heating_balance_points=heating_balance_points,
            cooling_balance_points=cooling_balance_points,
            degree_day_method=degree_day_method,
            percent_hourly_coverage_per_day=percent_hourly_coverage_per_day,
            use_mean_daily_values=use_mean_daily_values,
        ))
        temp_agg_column_renames.update({
            ('temp', 'degree_day_columns'): 'degree_day_columns',
        })

    if data_quality:
        temp_agg_funcs.extend([
            ('not_null', 'count'),
            ('null', lambda x: x.isnull().sum()),
        ])
        temp_agg_column_renames.update({
            ('temp', 'not_null'): 'temperature_not_null',
            ('temp', 'null'): 'temperature_null'
        })

    if temperature_mean:
        temp_agg_funcs.extend([('mean', 'mean')])
        temp_agg_column_renames.update({('temp', 'mean'): 'temperature_mean'})

    dfs_to_merge = [meter_data.value.to_frame('meter_value')]

    # aggregate temperatures
    temp_df = temperature_data.to_frame('temp')
    temp_groups = _matching_groups(meter_data.index, temp_df)
    dfs_to_merge.append(temp_groups.agg({'temp': temp_agg_funcs}))

    df = pd.concat(dfs_to_merge, axis=1).rename(
        columns=temp_agg_column_renames
    )

    # convert to average daily values.
    if use_mean_daily_values and meter_data.index.freq is None:
        df['meter_value'] = df.meter_value / day_counts(df.meter_value)

    # expand degree_day_columns
    if 'degree_day_columns' in df:
        df = pd.concat([
            df.drop(['degree_day_columns'], axis=1),
            df['degree_day_columns'].apply(pd.Series)
        ], axis=1)

    return df


def billing_as_daily(df, value_col='value'):
    ''' Convert billing period data to daily using daily period averages.

    Parameters
    ----------
    df : :any:`pandas.DataFrame`
        Data to convert from billing to daily frequency.
    value_col : :any:`str`, optional
        Name of value column of which the mean will be taken.

    Returns
    -------
    df : :any:`pandas.DataFrame`
        Daily-frequency data with average daily billing usage for each day.
    '''
    # TODO(philngo): incorporate this directly into merge_temperature_data

    # dont affect the original data
    df = df.copy()

    # convert to period mean
    df[value_col].iloc[:-1] = df[value_col].iloc[:-1] / (df.iloc[1:].index - df.iloc[:-1].index).days

    # last value is not kept.
    df[value_col].iloc[-1] = np.nan

    df = df.resample('D').ffill()

    return df


def day_counts(series):
    '''Days between index datetime values as a :any:`pandas.Series`.

    Parameters
    ----------
    series : :any:`pandas.Series` with :any:`pandas.DatetimeIndex`
        A series for which to get day counts.
    Returns
    -------
    day_counts : :any:`pandas.Series`
        A :any:`pandas.Series` with counts of days between periods. Counts are
        given on start dates of periods.
    '''
    # TODO(philngo): incorporate this directly into merge_temperature_data

    # dont affect the original data
    series = series.copy()

    series.iloc[:-1] = (series.index[1:] - series.index[:-1]).days
    series.iloc[-1] = np.nan
    return series


def get_baseline_data(data, start=None, end=None, max_days=365):
    ''' Filter down to baseline period data.

    Parameters
    ----------
    data : :any:`pandas.DataFrame` or :any:`pandas.Series`
        The data to filter to baseline data. This data will be filtered down
        to an acceptable baseline period according to the dates passed as
        `start` and `end`, or the maximum period specified with `max_days`.
    start : datetime.datetime
        A timezone-aware datetime that represents the earliest allowable start
        date for the baseline data. The stricter of this or `max_days` is used
        to determine the earliest allowable baseline period date.
    end : datetime.datetime
        A timezone-aware datetime that represents the latest allowable end
        date for the baseline data, i.e., the latest date for which data is
        available before the intervention begins.
    max_days : int
        The maximum length of the period. Ignored if `end` is not set.
        The stricter of this or `start` is used to determine the earliest
        allowable baseline period date.

    Returns
    -------
    baseline_data : :any:`pandas.DataFrame`
        Data for only the specified baseline period.
    '''

    if start is None:
        # py datetime min/max are out of range of pd.Timestamp min/max
        start = pytz.UTC.localize(pd.Timestamp.min)

    if end is None:
        end = pytz.UTC.localize(pd.Timestamp.max)
    else:
        if max_days is not None:
            min_start = end - timedelta(days=max_days)
            if start < min_start:
                start = min_start

    baseline_data = data[start:end]

    # TODO(philngo): report warnings about gaps at end

    # TODO(philngo): should this be a minimum of two or three datapoints to
    # prevent fitting errors?
    if baseline_data.empty:
        raise NoBaselineDataError()

    return baseline_data


def get_reporting_data(data, start=None, end=None, max_days=365):
    ''' Filter down to reporting period data.

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
    reporting_data : :any:`pandas.DataFrame`
        Data for only the specified reporting period.
    '''

    # TODO(philngo): report warnings about gaps at start

    if end is None:
        # py datetime min/max are out of range of pd.Timestamp min/max
        end = pytz.UTC.localize(pd.Timestamp.max)

    if start is None:
        start = pytz.UTC.localize(pd.Timestamp.min)
    else:
        if max_days is not None:
            max_end = start + timedelta(days=max_days)
            if end > max_end:
                end = max_end

    reporting_data = data[start:end]

    # TODO(philngo): should this be a minimum of two or three datapoints to
    # prevent fitting errors?
    if reporting_data.empty:
        raise NoReportingDataError()

    return reporting_data
