import numpy as np
import pandas as pd
from .api import (
    EEMeterWarning,
)
import statsmodels.formula.api as smf
import traceback
pd.options.mode.chained_assignment = None


def get_calendar_year_coverage_warning(baseline_data_segmented):

    warnings = []
    unique_models = baseline_data_segmented.model_id.unique()
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    if len(captured_months) < 12:
        warnings = [EEMeterWarning(
                qualified_name=('eemeter.caltrack_hourly.'
                                'incomplete_calendar_year_coverage'),
                description=(
                        'Data does not cover full calendar year. '
                        '{} Missing monthly models: {}'
                        .format(12 - len(captured_months),
                                [month for month in range(1, 13)
                                if month not in captured_months])
                    ),
                data={'num_missing_months': 12 - len(captured_months),
                      'missing_months': [month for month in range(1, 13)
                                         if month not in captured_months]}
                      )]
    return warnings


def get_hourly_coverage_warning(
        data, baseline_months, model_id, min_fraction_daily_coverage=0.9,):

    warnings = []
    summary = data.assign(total_days=data.index.days_in_month)
    summary = summary.groupby(summary.index.month) \
        .agg({'meter_value': len, 'model_id': max, 'total_days': max})
    summary['hourly_coverage'] = summary.meter_value / \
        (summary.total_days * 24)

    for month in baseline_months:
        row = summary.loc[summary.index == month]
        if (len(row.index) == 0):
            warnings.extend([EEMeterWarning(
                    qualified_name=('eemeter.caltrack_hourly.'
                                    'no_baseline_data_for_month'),
                    description=(
                            'No data for one of the baseline months. '
                            'Month {}'
                            .format(month)
                            ),
                    data={'model_id': model_id,
                          'month': month,
                          'hourly_coverage': 0}
                          )])
            continue

        if (row.hourly_coverage.values[0] < min_fraction_daily_coverage):
            warnings.extend([EEMeterWarning(
                    qualified_name=('eemeter.caltrack_hourly.'
                                    'insufficient_hourly_coverage'),
                    description=(
                            'Data for this model does not meet the minimum '
                            'hourly sufficiency criteria. '
                            'Month {}: Coverage: {}'
                            .format(month, row.hourly_coverage.values[0])
                            ),
                    data={'model_id': model_id,
                          'month': month,
                          'total_days': row.total_days.values[0],
                          'hourly_coverage': row.hourly_coverage.values[0]}
                          )])

    return warnings


def assign_baseline_periods(data, baseline_type):

    baseline_data = data.copy()
    baseline_data_segmented = pd.DataFrame()
    warnings = []
    valid_baseline_types = ['one_month',
                            'three_month',
                            'three_month_weighted',
                            'single', ]
    if baseline_type not in valid_baseline_types:
        raise ValueError('Invalid baseline type: %s' % (baseline_type))

    for column in ['meter_value', 'temperature_mean']:
        if column not in data.columns:
            raise ValueError('Data does not include columns: {}'
                             .format(column))

    if baseline_type == 'one_month':
        baseline_data_segmented = baseline_data.copy()
        baseline_data_segmented['weight'] = 1
        baseline_data_segmented['model_id'] = \
            [tuple([x]) for x in baseline_data_segmented.index.month]
    elif baseline_type in ['three_month', 'three_month_weighted']:
        unique_months = pd.Series(
                baseline_data.index
                .map(lambda x: x.month)
                .unique().values) \
                .map(lambda x: (x,))
        months = pd.DataFrame(unique_months, columns=['model_id'])

        def shoulder_months(month):
            if month == 1:
                return (12, 1, 2)
            elif month == 12:
                return (11, 12, 1)
            else:
                return (month - 1, month, month + 1)

        months.loc[:, 'baseline'] = months.model_id \
            .map(lambda x: shoulder_months(x[0]))
        for i, month in months.iterrows():
            this_df = baseline_data \
                    .loc[baseline_data.index.month.isin(month.baseline)]
            this_df.loc[:, 'model_id'] = \
                [month.model_id] * len(this_df.index)
            warnings.extend(get_hourly_coverage_warning(
                    this_df, month.baseline, month.model_id))

            this_df['weight'] = 1
            if baseline_type == 'three_month_weighted':
                this_df.loc[
                        [x[0] not in x[1] for x in
                         zip(this_df.index.month, this_df.model_id)],
                        'weight'] = 0.5
            baseline_data_segmented = baseline_data_segmented.append(
                    this_df, sort=False)
    elif baseline_type == 'single':
        baseline_data_segmented = baseline_data.copy()
        baseline_data_segmented['weight'] = 1
        baseline_data_segmented['model_id'] = \
            [tuple(range(1, 13))
             for j in range(len(baseline_data_segmented.index))]

#    baseline_data_segmented = baseline_data_segmented.reset_index()
    warnings.extend(get_calendar_year_coverage_warning(
            baseline_data_segmented))
    return baseline_data_segmented, warnings


def get_feature_hour_of_week(data):
    warnings = []
    feature_hour_of_week = \
        pd.DataFrame(data.index.dayofweek * 24 + (data.index.hour+1),
                     index=data.index) \
        .rename(columns={'start': 'hour_of_week'})
    feature_hour_of_week["hour_of_week"] = \
        feature_hour_of_week["hour_of_week"].astype('category')
    captured_hours = feature_hour_of_week.hour_of_week.unique()
    missing_hours = [hour for hour in range(1, 169)
                     if hour not in captured_hours]
    if sorted(feature_hour_of_week.hour_of_week.unique()) != \
            [x for x in range(1, 169)]:
                warnings = [EEMeterWarning(
                        qualified_name=('eemeter.caltrack_hourly.'
                                        'missing_hours_of_week'),
                        description=(
                                'Data does not include all hours of week. '
                                'Missing hours of week: {}'
                                .format(missing_hours)
                                ),
                        data={'num_missing_hours': 168 - len(captured_hours),
                              'missing_hours': missing_hours}
                        )]
    return feature_hour_of_week, warnings


def get_fit_failed_occupancy_model_warning(model_id):
    warning = [EEMeterWarning(
        qualified_name='eemeter.caltrack_hourly.failed_occupancy_model',
        description=(
            'Error encountered in statsmodels.formula.api.wls method '
            'for occupancy model id: {}'.format(model_id)
        ),
        data={'model_id': model_id,
              'traceback': traceback.format_exc()}
    )]
    return warning


def ishighusage(df, threshold, residual_col='residuals'):
    df = df.rename(columns={residual_col: 'residuals'})

    return int(sum(df.residuals > 0) / len(df.residuals) > threshold)


def get_single_feature_occupancy(data, threshold):
    warnings = []

    # TODO: replace with design matrix function
    model_data = data.assign(
            cdd_65=data.temperature_mean.map(lambda x: max(0, x-65)),
            hdd_50=data.temperature_mean.map(lambda x: max(0, 50-x)))

    try:
        model_occupancy = smf.wls(formula='meter_value ~ cdd_65 + hdd_50',
                                  data=model_data,
                                  weights=model_data.weight)
    except Exception as e:
        warnings.extend(
                get_fit_failed_occupancy_model_warning(data.model_id[0]))
        return pd.DataFrame(), pd.DataFrame(), warnings

    model_data = model_data.merge(
            pd.DataFrame({'residuals': model_occupancy.fit().resid}),
            left_index=True, right_index=True)

    # TODO: replace with design matrix function
    feature_hour_of_week, warnings = get_feature_hour_of_week(data)
    model_data = model_data.merge(feature_hour_of_week,
                                  left_index=True, right_index=True)

    lookup_occupancy = pd.DataFrame({
            'occupancy': model_data.groupby(['hour_of_week'])
            .apply(lambda x: ishighusage(x, threshold))}) \
        .reset_index()

    feature_occupancy = model_data.reset_index() \
        .merge(lookup_occupancy,
               left_on=['hour_of_week'],
               right_on=['hour_of_week']) \
        .set_index('start').sort_index() \
        .loc[:, ['occupancy']]

    return feature_occupancy, lookup_occupancy, warnings


def get_missing_model_id_warning(columns):
    warning = [EEMeterWarning(
        qualified_name='eemeter.caltrack_hourly.missing_model_id',
        description=(
            'Warning: Model ID is missing - this function will be run '
            'with all of the data in a single model'
        ),
        data={'dataframe_columns': columns}
    )]
    return warning


def get_missing_weight_column_warning(columns):
    warning = [EEMeterWarning(
        qualified_name='eemeter.caltrack_hourly.missing_weight_column',
        description=(
            'Warning: Weight column is missing - this function will be run '
            'without any data weights'
        ),
        data={'dataframe_columns': columns}
    )]
    return warning


def get_feature_occupancy(data, threshold=0.65):
    warnings = []
    feature_occupancy = pd.DataFrame()
    lookup_occupancy = pd.DataFrame()

    if 'model_id' not in data.columns:
        warnings.extend(get_missing_model_id_warning(data.columns))
        data['model_id'] = [(1,)] * len(data.index)
    if 'weight' not in data.columns:
        warnings.extend(get_missing_weight_column_warning(data.columns))
        data['weight'] = 1

    unique_models = data.model_id.unique()
    for model_id in unique_models:
        this_data = data.loc[data.model_id == model_id]
        this_feature_occupancy, this_lookup_occupancy, this_warnings = \
            get_single_feature_occupancy(this_data, threshold)

        this_feature_occupancy['model_id'] = [model_id] * \
            len(this_feature_occupancy.index)
        this_lookup_occupancy['model_id'] = [model_id] * \
            len(this_lookup_occupancy.index)

        feature_occupancy = feature_occupancy.append(this_feature_occupancy)
        lookup_occupancy = lookup_occupancy.append(this_lookup_occupancy)
        warnings.extend(this_warnings)

    return feature_occupancy, lookup_occupancy, warnings


def get_design_matrix(data, functions):
    design_matrix = pd.DataFrame()
    feature_parameters = {}
    warnings = []
    return design_matrix, feature_parameters, warnings
