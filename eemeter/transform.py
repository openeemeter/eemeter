#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2018 Open Energy Efficiency, Inc.

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
from datetime import datetime, timedelta
from functools import partial

import numpy as np
import pandas as pd
import pytz

from .exceptions import NoBaselineDataError, NoReportingDataError
from .warnings import EEMeterWarning


__all__ = (
    "as_freq",
    "day_counts",
    "get_baseline_data",
    "get_reporting_data",
    "remove_duplicates",
    "overwrite_partial_rows_with_nan",
)


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


def as_freq(data_series, freq, atomic_freq="1 Min", series_type="cumulative"):
    """Resample data to a different frequency.

    This method can be used to upsample or downsample meter data. The
    assumption it makes to do so is that meter data is constant and averaged
    over the given periods. For instance, to convert billing-period data to
    daily data, this method first upsamples to the atomic frequency
    (1 minute freqency, by default), "spreading" usage evenly across all
    minutes in each period. Then it downsamples to hourly frequency and
    returns that result. With instantaneous series, the data is copied to all
    contiguous time intervals and the mean over `freq` is returned.

    **Caveats**:

     - This method gives a fair amount of flexibility in
       resampling as long as you are OK with the assumption that usage is
       constant over the period (this assumption is generally broken in
       observed data at large enough frequencies, so this caveat should not be
       taken lightly).

    Parameters
    ----------
    data_series : :any:`pandas.Series`
        Data to resample. Should have a :any:`pandas.DatetimeIndex`.
    freq : :any:`str`
        The frequency to resample to. This should be given in a form recognized
        by the :any:`pandas.Series.resample` method.
    atomic_freq : :any:`str`, optional
        The "atomic" frequency of the intermediate data form. This can be
        adjusted to a higher atomic frequency to increase speed or memory
        performance.
    series_type : :any:`str`, {'cumulative', ‘instantaneous’},
        default 'cumulative'
        Type of data sampling. 'cumulative' data can be spread over smaller
        time intervals and is aggregated using addition (e.g. meter data).
        'instantaneous' data is copied (not spread) over smaller time intervals
        and is aggregated by averaging (e.g. weather data).

    Returns
    -------
    resampled_data : :any:`pandas.Series`
        Data resampled to the given frequency.
    """
    # TODO(philngo): make sure this complies with CalTRACK 2.2.2.1
    if not isinstance(data_series, pd.Series):
        raise ValueError(
            "expected series, got object with class {}".format(data_series.__class__)
        )
    if data_series.empty:
        return data_series
    series = remove_duplicates(data_series)
    target_freq = pd.Timedelta(atomic_freq)
    timedeltas = (series.index[1:] - series.index[:-1]).append(
        pd.TimedeltaIndex([pd.NaT])
    )

    if series_type == "cumulative":
        spread_factor = target_freq.total_seconds() / timedeltas.total_seconds()
        series_spread = series * spread_factor
        atomic_series = series_spread.asfreq(atomic_freq, method="ffill")
        resampled = atomic_series.resample(freq).sum()
        resampled_with_nans = atomic_series.resample(freq).mean()
        resampled = resampled[resampled_with_nans.notnull()].reindex(resampled.index)

    elif series_type == "instantaneous":
        atomic_series = series.asfreq(atomic_freq, method="ffill")
        resampled = atomic_series.resample(freq).mean()

    resampled.iloc[-1] = np.nan
    return resampled


def day_counts(index):
    """Days between DatetimeIndex values as a :any:`pandas.Series`.

    Parameters
    ----------
    index : :any:`pandas.DatetimeIndex`
        The index for which to get day counts.

    Returns
    -------
    day_counts : :any:`pandas.Series`
        A :any:`pandas.Series` with counts of days between periods. Counts are
        given on start dates of periods.
    """
    # dont affect the original data
    index = index.copy()

    if len(index) == 0:
        return pd.Series([], index=index)

    timedeltas = (index[1:] - index[:-1]).append(pd.TimedeltaIndex([pd.NaT]))
    timedelta_days = timedeltas.total_seconds() / (60 * 60 * 24)

    return pd.Series(timedelta_days, index=index)


def _make_baseline_warnings(
    end_inf, start_inf, data_start, data_end, start_limit, end_limit
):
    warnings = []
    # warn if there is a gap at end
    if not end_inf and data_end < end_limit:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_baseline_data.gap_at_baseline_end",
                description=(
                    "Data does not have coverage at requested baseline end date."
                ),
                data={
                    "requested_end": end_limit.isoformat(),
                    "data_end": data_end.isoformat(),
                },
            )
        )
    # warn if there is a gap at start
    if not start_inf and start_limit < data_start:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_baseline_data.gap_at_baseline_start",
                description=(
                    "Data does not have coverage at requested baseline start date."
                ),
                data={
                    "requested_start": start_limit.isoformat(),
                    "data_start": data_start.isoformat(),
                },
            )
        )
    return warnings


def get_baseline_data(
    data,
    start=None,
    end=None,
    max_days=365,
    allow_billing_period_overshoot=False,
    ignore_billing_period_gap_for_day_count=False,
):
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
    max_days : :any:`int`, default 365
        The maximum length of the period. Ignored if `end` is not set.
        The stricter of this or `start` is used to determine the earliest
        allowable baseline period date.
    allow_billing_period_overshoot : :any:`bool`, default False
        If True, count `max_days` from the end of the last billing data period
        that ends before the `end` date, rather than from the exact `end` date.
        Otherwise use the exact `end` date as the cutoff.
    ignore_billing_period_gap_for_day_count : :any:`bool`, default False
        If True, instead of going back `max_days` from either the
        `end` date or end of the last billing period before that date (depending
        on the value of the `allow_billing_period_overshoot` setting) and
        excluding the last period that began before that date, first check to
        see if excluding or including that period gets closer to a total of
        `max_days` of data.

        For example, with `max_days=365`, if an exact 365 period would targeted
        Feb 15, but the billing period went from Jan 20 to Feb 20, exclude that
        period for a total of ~360 days of data, because that's closer to 365
        than ~390 days, which would be the total if that period was included.
        If, on the other hand, if that period started Feb 10 and went to Mar 10,
        include the period, because ~370 days of data is closer to than ~340.

    Returns
    -------
    baseline_data, warnings : :any:`tuple` of (:any:`pandas.DataFrame` or :any:`pandas.Series`, :any:`list` of :any:`eemeter.EEMeterWarning`)
        Data for only the specified baseline period and any associated warnings.
    """
    if max_days is not None:
        if start is not None:
            raise ValueError(  # pragma: no cover
                "If max_days is set, start cannot be set: start={}, max_days={}.".format(
                    start, max_days
                )
            )

    start_inf = False
    if start is None:
        # py datetime min/max are out of range of pd.Timestamp min/max
        start_target = pytz.UTC.localize(pd.Timestamp.min)
        start_inf = True
    else:
        start_target = start

    end_inf = False
    if end is None:
        end_limit = pytz.UTC.localize(pd.Timestamp.max)
        end_inf = True
    else:
        end_limit = end

    # copying prevents setting on slice warnings
    data_before_end_limit = data[:end_limit].copy()

    if ignore_billing_period_gap_for_day_count:
        end_limit = data_before_end_limit.index.max()

    if not end_inf and max_days is not None:
        start_target = end_limit - timedelta(days=max_days)

    if allow_billing_period_overshoot:
        # adjust start limit to get a selection closest to max_days
        # also consider ffill for get_loc method - always picks previous
        try:
            loc = data_before_end_limit.index.get_loc(start_target, method="nearest")
        except (KeyError, IndexError):  # pragma: no cover
            baseline_data = data_before_end_limit
            start_limit = start_target
        else:
            start_limit = data_before_end_limit.index[loc]
            baseline_data = data_before_end_limit[start_limit:].copy()

    else:
        # use hard limit for baseline start
        start_limit = start_target
        baseline_data = data_before_end_limit[start_limit:].copy()

    if baseline_data.dropna().empty:
        raise NoBaselineDataError()

    baseline_data.iloc[-1] = np.nan

    data_end = data.index.max()
    data_start = data.index.min()
    return (
        baseline_data,
        _make_baseline_warnings(
            end_inf, start_inf, data_start, data_end, start_limit, end_limit
        ),
    )


def _make_reporting_warnings(
    end_inf, start_inf, data_start, data_end, start_limit, end_limit
):
    warnings = []
    # warn if there is a gap at end
    if not end_inf and data_end < end_limit:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_reporting_data.gap_at_reporting_end",
                description=(
                    "Data does not have coverage at requested reporting end date."
                ),
                data={
                    "requested_end": end_limit.isoformat(),
                    "data_end": data_end.isoformat(),
                },
            )
        )
    # warn if there is a gap at start
    if not start_inf and start_limit < data_start:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.get_reporting_data.gap_at_reporting_start",
                description=(
                    "Data does not have coverage at requested reporting start date."
                ),
                data={
                    "requested_start": start_limit.isoformat(),
                    "data_start": data_start.isoformat(),
                },
            )
        )
    return warnings


def get_reporting_data(
    data,
    start=None,
    end=None,
    max_days=365,
    allow_billing_period_overshoot=False,
    ignore_billing_period_gap_for_day_count=False,
):
    """ Filter down to reporting period data.

    Parameters
    ----------
    data : :any:`pandas.DataFrame` or :any:`pandas.Series`
        The data to filter to reporting data. This data will be filtered down
        to an acceptable reporting period according to the dates passed as
        `start` and `end`, or the maximum period specified with `max_days`.
    start : :any:`datetime.datetime`
        A timezone-aware datetime that represents the earliest allowable start
        date for the reporting data, i.e., the earliest date for which data is
        available after the intervention begins.
    end : :any:`datetime.datetime`
        A timezone-aware datetime that represents the latest allowable end
        date for the reporting data. The stricter of this or `max_days` is used
        to determine the latest allowable reporting period date.
    max_days : :any:`int`, default 365
        The maximum length of the period. Ignored if `start` is not set.
        The stricter of this or `end` is used to determine the latest
        allowable reporting period date.
    allow_billing_period_overshoot : :any:`bool`, default False
        If True, count `max_days` from the start of the first billing data period
        that starts after the `start` date, rather than from the exact `start` date.
        Otherwise use the exact `start` date as the cutoff.
    ignore_billing_period_gap_for_day_count : :any:`bool`, default False
        If True, instead of going forward `max_days` from either the
        `start` date or the `start` of the first billing period after that date
        (depending on the value of the `allow_billing_period_overshoot` setting)
        and excluding the first period that ended after that date, first check
        to see if excluding or including that period gets closer to a total of
        `max_days` of data.

        For example, with `max_days=365`, if an exact 365 period would targeted
        Feb 15, but the billing period went from Jan 20 to Feb 20, include that
        period for a total of ~370 days of data, because that's closer to 365
        than ~340 days, which would be the total if that period was excluded.
        If, on the other hand, if that period started Feb 10 and went to Mar 10,
        exclude the period, because ~360 days of data is closer to than ~390.

    Returns
    -------
    reporting_data, warnings : :any:`tuple` of (:any:`pandas.DataFrame` or :any:`pandas.Series`, :any:`list` of :any:`eemeter.EEMeterWarning`)
        Data for only the specified reporting period and any associated warnings.
    """
    if max_days is not None:
        if end is not None:
            raise ValueError(  # pragma: no cover
                "If max_days is set, end cannot be set: end={}, max_days={}.".format(
                    end, max_days
                )
            )

    start_inf = False
    if start is None:
        # py datetime min/max are out of range of pd.Timestamp min/max
        start_limit = pytz.UTC.localize(pd.Timestamp.min)
        start_inf = True
    else:
        start_limit = start

    end_inf = False
    if end is None:
        end_target = pytz.UTC.localize(pd.Timestamp.max)
        end_inf = True
    else:
        end_target = end

    # copying prevents setting on slice warnings
    data_after_start_limit = data[start_limit:].copy()

    if ignore_billing_period_gap_for_day_count:
        start_limit = data_after_start_limit.index.min()

    if not start_inf and max_days is not None:
        end_target = start_limit + timedelta(days=max_days)

    if allow_billing_period_overshoot:
        # adjust start limit to get a selection closest to max_days
        # also consider bfill for get_loc method - always picks next
        try:
            loc = data_after_start_limit.index.get_loc(end_target, method="nearest")
        except (KeyError, IndexError):  # pragma: no cover
            reporting_data = data_after_start_limit
            end_limit = end_target
        else:
            end_limit = data_after_start_limit.index[loc]
            reporting_data = data_after_start_limit[:end_limit].copy()

    else:
        # use hard limit for baseline start
        end_limit = end_target
        reporting_data = data_after_start_limit[:end_limit].copy()

    if reporting_data.dropna().empty:
        raise NoReportingDataError()

    reporting_data.iloc[-1] = np.nan

    data_end = data.index.max()
    data_start = data.index.min()
    return (
        reporting_data,
        _make_reporting_warnings(
            end_inf, start_inf, data_start, data_end, start_limit, end_limit
        ),
    )
