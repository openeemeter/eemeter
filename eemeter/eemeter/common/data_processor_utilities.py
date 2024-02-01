from eemeter.eemeter.warnings import EEMeterWarning
import numpy as np
import pandas as pd


def remove_duplicates(df_or_series):
    """Remove duplicate rows or values by keeping the first of each duplicate.

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

def clean_caltrack_billing_data(data, source_interval):
    # check for empty data
    if data["value"].dropna().empty:
        return data[:0]

    if source_interval.startswith("billing"):
        diff = list((data.index[1:] - data.index[:-1]).days)
        filter_ = pd.Series(diff + [np.nan], index=data.index)

        # CalTRACK 2.2.3.4, 2.2.3.5
        if source_interval == "billing_monthly":
            data = data[
                (filter_ <= 35) & (filter_ >= 25)  # keep these, inclusive
            ].reindex(data.index)

        # CalTRACK 2.2.3.4, 2.2.3.5
        if source_interval == "billing_bimonthly":
            data = data[
                (filter_ <= 70) & (filter_ >= 25)  # keep these, inclusive
            ].reindex(data.index)

        # CalTRACK 2.2.3.1
        """
        Adds estimate to subsequent read if there aren't more than one estimate in a row
        and then removes the estimated row.

        Input:
        index   value   estimated
        1       2       False
        2       3       False
        3       5       True
        4       4       False
        5       6       True
        6       3       True
        7       4       False
        8       NaN     NaN

        Output:
        index   value
        1       2
        2       3
        4       9
        5       NaN
        7       7
        8       NaN
        """
        add_estimated = []
        remove_estimated_fixed_rows = []
        orig_data = data.copy()
        if "estimated" in data.columns:
            data["unestimated_value"] = (
                data[:-1].value[(data[:-1].estimated == False)].reindex(data.index)
            )
            data["estimated_value"] = (
                data[:-1].value[(data[:-1].estimated)].reindex(data.index)
            )
            for i, (index, row) in enumerate(data[:-1].iterrows()):
                # ensures there is a prev_row and previous row value is null
                if i > 0 and pd.isnull(prev_row["unestimated_value"]):
                    # current row value is not null
                    add_estimated.append(prev_row["estimated_value"])
                    if not pd.isnull(row["unestimated_value"]):
                        # get all rows that had only estimated reads that will be
                        # added to the subsequent row meaning this row
                        # needs to be removed
                        remove_estimated_fixed_rows.append(prev_index)
                else:
                    add_estimated.append(0)
                prev_row = row
                prev_index = index
            add_estimated.append(np.nan)
            data["value"] = data["unestimated_value"] + add_estimated
            data = data[~data.index.isin(remove_estimated_fixed_rows)]
            data = data[["value"]]  # remove the estimated column

    # check again for empty data
    if data.dropna().empty:
        return data[:0]

    return data


def as_freq(
    data_series,
    freq,
    atomic_freq="1 Min",
    series_type="cumulative",
    include_coverage=False,
):
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
    include_coverage: :any:`bool`,
        default `False`
        Option of whether to return a series with just the resampled values
        or a dataframe with a column that includes percent coverage of source data
        used for each sample.

    Returns
    -------
    resampled_data : :any:`pandas.Series` or :any:`pandas.DataFrame`
        Data resampled to the given frequency (optionally as a dataframe with a coverage column if `include_coverage` is used.
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
        resampled_with_nans = atomic_series.resample(freq).first()
        n_coverage = atomic_series.resample(freq).count()
        resampled = resampled[resampled_with_nans.notnull()].reindex(resampled.index)

    elif series_type == "instantaneous":
        atomic_series = series.asfreq(atomic_freq, method="ffill")
        resampled = atomic_series.resample(freq).mean()

    if resampled.index[-1] < series.index[-1]:
        # this adds a null at the end using the target frequency
        last_index = pd.date_range(resampled.index[-1], freq=freq, periods=2)[1:]
        resampled = (
            pd.concat([resampled, pd.Series(np.nan, index=last_index)])
            .resample(freq)
            .mean()
        )
    if include_coverage:
        n_total = resampled.resample(atomic_freq).count().resample(freq).count()
        resampled = resampled.to_frame("value")
        resampled["coverage"] = n_coverage / n_total
        return resampled
    else:
        return resampled


def downsample_and_clean_caltrack_daily_data(dataset, warnings):
    dataset = as_freq(dataset, "D", include_coverage=True)

    if dataset[dataset.coverage <= 0.5] is not None:
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.caltrack_sufficiency_criteria.missing_high_frequency_data",
                description=("More than 50% of the high frequency  data is missing."),
                data = (
                    dataset[dataset.coverage <= 0.5].index.to_list()
                )
            )
        )

    # CalTRACK 2.2.2.1 - interpolate with average of non-null values
    dataset.value[dataset.coverage > 0.5] = (
        dataset[dataset.coverage > 0.5].value / dataset[dataset.coverage > 0.5].coverage
    )

    return dataset[dataset.coverage > 0.5].reindex(dataset.index)[["value"]]


def clean_caltrack_billing_daily_data(data, source_interval, warnings):
    # billing data is cleaned but not resampled
    if source_interval.startswith("billing"):
        # CalTRACK 2.2.3.4, 2.2.3.5
        return clean_caltrack_billing_data(data, source_interval, warnings)

    # higher intervals like daily, hourly, 30min, 15min are
    # resampled (daily) or downsampled (hourly, 30min, 15min)
    elif source_interval == "daily":
        return data
    else:
        return downsample_and_clean_caltrack_daily_data(data, warnings)