from .transform import (
    overwrite_partial_rows_with_nan,
)

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


__all__ = (
    'iterate_segmented_dataset',
    'compute_temperature_features',
    'compute_time_features',
    'estimate_hour_of_week_occupancy',
    'fit_temperature_bins'
)


def iterate_segmented_dataset(data, segmentation=None):
    if segmentation is None:
        yield None, pd.merge(
            data, pd.DataFrame({'weight': 1}, index=data.index),
            left_index=True, right_index=True
        )  # add weight column
    else:
        for segment_name, segment_weights in segmentation.iteritems():
            segment_data = pd.merge(
                data, segment_weights.to_frame('weight'),
                left_index=True, right_index=True
            )[segment_weights > 0]  # take only non zero weights
            yield segment_name, segment_data


def _get_missing_hours_of_week_warning():
    pass


def compute_time_features(index, hour_of_week=True):
    if hour_of_week:
        time_features = pd.DataFrame({
            'hour_of_week': (
                index.dayofweek * 24 + (index.hour + 1)
            ).astype('category'),
        }, index=index)
    else:
        time_features = pd.DataFrame({}, index=True)

    # TODO: do something with this
    _get_missing_hours_of_week_warning()

    return time_features


def _matching_groups(index, df, tolerance):
    # convert index to df for use with merge_asof
    index_df = pd.DataFrame({'index_col': index}, index=index)

    # get a dataframe containing mean temperature
    #   1) merge by matching temperature to closest previous meter start date,
    #      up to tolerance limit, using merge_asof.
    #   2) group by meter_index, and take the mean, ignoring all columns except
    #      the temperature column.
    groups = pd.merge_asof(
        left=df, right=index_df, left_index=True, right_index=True,
        tolerance=tolerance
    ).groupby('index_col')
    return groups


def _degree_day_columns(
    heating_balance_points, cooling_balance_points, degree_day_method,
    percent_hourly_coverage_per_day, percent_hourly_coverage_per_billing_period,
    use_mean_daily_values,
):
    # TODO(philngo): can this be refactored to be a more general without losing
    # on performance?

    # Not used in CalTRACK 2.0
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

    # CalTRACK 2.2.2.3
    n_limit_daily = 24 * percent_hourly_coverage_per_day

    if degree_day_method == 'daily':
        def _compute_columns(temps):
            count = temps.shape[0]
            if count > 24:

                day_groups = np.floor(np.arange(count) / 24)
                daily_temps = temps.groupby(day_groups).agg(['mean', 'count'])
                n_limit_period = (percent_hourly_coverage_per_billing_period *
                    count)
                n_days_total = daily_temps.shape[0]

                # CalTrack 2.2.3.2
                if temps.notnull().sum() < n_limit_period:
                    daily_temps = daily_temps['mean'].iloc[0:0]
                else:
                    # CalTRACK 2.2.2.3
                    daily_temps = daily_temps['mean'][daily_temps['count'] > n_limit_daily]
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

                if count > n_limit_daily:
                    count_cols = {
                        'n_days_kept': 1,
                        'n_days_dropped': 0,
                    }
                    # CalTRACK 2.2.2.3
                    mean_temp = temps.mean()
                else:
                    count_cols = {
                        'n_days_kept': 0,
                        'n_days_dropped': 1,
                    }
                    mean_temp = np.nan

                # CalTrack 3.3.4.1.1
                cdd_cols = {
                    'cdd_%s' % bp: np.maximum(mean_temp - bp, 0)
                    for bp in cooling_balance_points
                }

                # CalTrack 3.3.5.1.1
                hdd_cols = {
                    'hdd_%s' % bp: np.maximum(bp - mean_temp, 0)
                    for bp in heating_balance_points
                }

            columns = count_cols
            columns.update(cdd_cols)
            columns.update(hdd_cols)
            return columns
    # TODO(philngo): option to ignore the count columns?

    agg_funcs = [('degree_day_columns', _compute_columns)]
    return agg_funcs


def compute_temperature_features(
    meter_data_index, temperature_data, heating_balance_points=None,
    cooling_balance_points=None, data_quality=False, temperature_mean=True,
    degree_day_method='daily', percent_hourly_coverage_per_day=0.5,
    percent_hourly_coverage_per_billing_period=0.9,
    use_mean_daily_values=True, tolerance=None, keep_partial_nan_rows=False
):
    ''' Compute temperature features from hourly temperature data using the
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
    '''
    if temperature_data.index.freq != 'H':
        raise ValueError(
            "temperature_data.index must have hourly frequency (freq='H')."
            " Found: {}"
            .format(temperature_data.index.freq)
        )

    if not temperature_data.index.tz:
        raise ValueError(
            "temperature_data.index must be timezone-aware. You can set it with"
            " temperature_data.tz_localize(...)."
        )

    if (
        meter_data_index.freq is None and
        meter_data_index.inferred_freq == 'H'
    ):
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
        if degree_day_method == 'hourly':
            pass
        elif degree_day_method == 'daily':
            if meter_data_index.freq == 'H':
                raise ValueError(
                    "degree_day_method='hourly' must be used with"
                    " hourly meter data. Found: 'daily'".format(degree_day_method)
                )
        else:
            raise ValueError('method not supported: {}'.format(degree_day_method))

    if pd.Timedelta(meter_data_index.freq) == pd.Timedelta('1H'):
        # special fast route for hourly data.
        df = temperature_data.to_frame('temperature_mean')\
            .reindex(meter_data_index)
        df = df.assign(**{
            'cdd_{}'.format(bp): np.maximum(df.temperature_mean - bp, 0)
            for bp in cooling_balance_points
        })
        df = df.assign(**{
            'hdd_{}'.format(bp): np.maximum(bp - df.temperature_mean, 0)
            for bp in heating_balance_points
        })
        if data_quality:
            df = df.assign(
                temperature_null=df.temperature_mean.isnull().astype(int),
                temperature_not_null=df.temperature_mean.notnull().astype(int),
            )
        if not temperature_mean:
            del df['temperature_mean']
    else:
        # daily/billing route
        # heating/cooling degree day aggregations. Needed for n_days fields as well.
        temp_agg_funcs.extend(_degree_day_columns(
            heating_balance_points=heating_balance_points,
            cooling_balance_points=cooling_balance_points,
            degree_day_method=degree_day_method,
            percent_hourly_coverage_per_day=percent_hourly_coverage_per_day,
            percent_hourly_coverage_per_billing_period=\
                percent_hourly_coverage_per_billing_period,
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

        # aggregate temperatures
        temp_df = temperature_data.to_frame('temp')
        temp_groups = _matching_groups(meter_data_index, temp_df, tolerance)
        temp_aggregations = temp_groups.agg({'temp': temp_agg_funcs})

        # expand temp aggregations by faking and delete the `meter_value` column.
        # I haven't yet figured out a way to avoid this and get the desired
        # structure and behavior. (philngo)
        meter_value = pd.DataFrame({'meter_value': 0}, index=meter_data_index)
        df = pd.concat([
            meter_value,
            temp_aggregations,
        ], axis=1).rename(columns=temp_agg_column_renames)
        del df['meter_value']

        if df.empty:
            if 'degree_day_columns' in df:
                column_defaults = {
                    column: []
                    for column in ['n_days_dropped', 'n_days_kept']
                }
                df = df.drop(['degree_day_columns'], axis=1) \
                    .assign(**column_defaults)
        else:
            # expand degree_day_columns
            if 'degree_day_columns' in df:
                df = pd.concat([
                    df.drop(['degree_day_columns'], axis=1),
                    df['degree_day_columns'].dropna().apply(pd.Series)
                ], axis=1)

    if not keep_partial_nan_rows:
        df = overwrite_partial_rows_with_nan(df)

    return df


def _estimate_hour_of_week_occupancy(model_data, threshold):
    usage_model = smf.wls(
        formula='meter_value ~ cdd_65 + hdd_50',
        data=model_data,
        weights=model_data.weight
    )

    model_data_with_residuals = model_data.merge(
        pd.DataFrame({'residuals': usage_model.fit().resid}),
        left_index=True, right_index=True)

    def _is_high_usage(df):
        n_positive_residuals = sum(df.residuals > 0)
        n_residuals = len(df.residuals)
        ratio_positive_residuals = n_positive_residuals / n_residuals
        return int(ratio_positive_residuals > threshold)

    return model_data_with_residuals \
        .groupby(['hour_of_week']) \
        .apply(_is_high_usage) \
        .rename('occupancy') \
        .reindex(range(1, 169))  # guarantee an index value for all hours


def estimate_hour_of_week_occupancy(data, segmentation=None, threshold=0.65):
    occupancy_lookups = {}
    segmented_datasets = iterate_segmented_dataset(data, segmentation)
    for segment_name, segmented_data in segmented_datasets:
        hour_of_week_occupancy = _estimate_hour_of_week_occupancy(
            segmented_data, threshold)
        column = 'occupancy' if segment_name is None else segment_name
        occupancy_lookups[column] = hour_of_week_occupancy
    return pd.DataFrame(occupancy_lookups)


def fit_temperature_bins(temperature_data, default_bins=):
    pass  # TODO


def validate_temperature_bins(
    data, default_bins, min_temperature_count=20
):
    bin_endpoints_valid = [-np.inf] + default_bins + [np.inf]
    temperature_data = data.loc[:, ['temperature_mean']]

    for i in range(1, len(bin_endpoints_valid)-1):  # not pythonic
        temperature_data['bin'] = pd.cut(
            temperature_data.temperature_mean, bins=bin_endpoints_valid
        )
        bins_default = [
            pd.Interval(
                bin_endpoints_valid[i],
                bin_endpoints_valid[i+1],
                closed='right'
            )
            for i in range(len(bin_endpoints_valid)-1)  # using i again?
        ]

        temperature_data.bin = temperature_data.bin \
            .cat.set_categories(bins_default)

        temperature_summary = temperature_data.groupby('bin') \
            .count().sort_index().reset_index()

        if i == 1:
            temperature_summary_original = temperature_summary.copy()

        first_bin_below_threshold = \
            temperature_summary.temperature_mean.iloc[0] < \
            min_temperature_count
        first_bin_null = np.isnan(temperature_summary.temperature_mean.iloc[0])

        if first_bin_below_threshold or first_bin_null:
            bin_endpoints_valid.remove(temperature_summary.iloc[0].bin.right)

        last_bin_below_threshold = \
            temperature_summary.temperature_mean.iloc[-1] < \
            min_temperature_count
        last_bin_null = np.isnan(temperature_summary.temperature_mean.iloc[-1])

        if last_bin_below_threshold or last_bin_null:
            bin_endpoints_valid.remove(temperature_summary.iloc[-1].bin.left)

    bin_endpoints_valid = [
        x for x in bin_endpoints_valid
        if x not in [-np.inf, np.inf]
    ]

    return temperature_summary_original, bin_endpoints_valid


# def get_single_feature_occupancy(data, threshold):
#     warnings = []
#
#     # TODO: replace with design matrix function
#     model_data = data.assign(
#         cdd_65=np.maximum(data.temperature_mean - 65, 0),

#         hdd_50=np.maximum(50 - data.temperature_mean, 0),
#     )
#
#     try:
#         model_occupancy = smf.wls(
#             formula='meter_value ~ cdd_65 + hdd_50',
#             data=model_data,
#             weights=model_data.weight
#         )
#     except Exception as e:
#         warnings.extend(
#             get_fit_failed_model_warning(
#                 data.model_id[0], 'occupancy_model'))
#         return None, pd.DataFrame(), warnings
#
#     model_data = model_data.merge(
#         pd.DataFrame({'residuals': model_occupancy.fit().resid}),
#         left_index=True, right_index=True)
#
#     # TODO: replace with design matrix function
#     feature_hour_of_week, parameters, warnings = get_feature_hour_of_week(data)
#     model_data = model_data.merge(
#         feature_hour_of_week, left_index=True, right_index=True)
#
#     def _is_high_usage(df, threshold, residual_col='residuals'):
#         df = df.rename(columns={residual_col: 'residuals'})
#         return int(sum(df.residuals > 0) / len(df.residuals) > threshold)
#
#     occupancy_lookup = pd.DataFrame({
#         'occupancy': model_data.groupby(['hour_of_week'])
#             .apply(lambda x: _is_high_usage(x, threshold))
#     }).reset_index()
#
#     return model_occupancy, occupancy_lookup, warnings
#
#
# def get_feature_occupancy(
#     data, mode='fit', threshold=0.65, occupancy_lookup=None, **kwargs
# ):
#     occupancy_warnings = []
#     occupancy_models = {}
#
#     data_verified, warnings = handle_unsegmented_timeseries(data)
#     occupancy_warnings.extend(warnings)
#
#     if mode == 'fit':
#         occupancy_lookup = pd.DataFrame()
#         unique_models = data_verified.model_id.unique()
#         for model_id in unique_models:
#             this_data = data_verified.loc[data_verified.model_id == model_id]
#             this_model, this_occupancy_lookup, this_warnings = \
#                 get_single_feature_occupancy(this_data, threshold)
#             this_occupancy_lookup['model_id'] = [model_id] * \
#                 len(this_occupancy_lookup.index)
#
#             occupancy_lookup = occupancy_lookup.append(
#                 this_occupancy_lookup, sort=False)
#             occupancy_warnings.extend(this_warnings)
#             occupancy_models[model_id] = this_model
#
#     if len(occupancy_lookup.index) == 0:
#         return pd.DataFrame(), {}, occupancy_warnings
#
#     feature_hour_of_week, parameters, warnings = \
#         get_feature_hour_of_week(data_verified)
#
#     feature_occupancy = data_verified.reset_index().merge(
#         feature_hour_of_week,
#         left_on=['start', 'model_id'],
#         right_on=['start', 'model_id']
#     ).merge(
#         occupancy_lookup,
#         left_on=['model_id', 'hour_of_week'],
#         right_on=['model_id', 'hour_of_week']
#     ).loc[:, ['start', 'model_id', 'occupancy']].set_index('start')
#
#     occupancy_parameters = {
#         'mode': 'predict',
#         'occupancy_models': occupancy_models,
#         'threshold': threshold,
#         'occupancy_lookup': occupancy_lookup
#     }
#     return feature_occupancy, occupancy_parameters, occupancy_warnings
