import numpy as np
import pandas as pd
from .api import (
    EEMeterWarning,
)
pd.options.mode.chained_assignment = None


def get_calendar_year_coverage_warning(baseline_data_segmented):

    warnings = []
    unique_models = baseline_data_segmented.model_months.unique()
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
        data, baseline_months, model_months, min_fraction_daily_coverage=0.9,):

    warnings = []
    summary = data.reset_index()
    summary = summary.groupby(summary.start.map(lambda x: x.month)) \
        .agg({'meter_value': len, 'start': max, 'model_months': max})
    summary['total_days'] = summary.apply(lambda x: x.start.days_in_month,
                                          axis=1)
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
                    data={'model_months': model_months,
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
                    data={'model_months': model_months,
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
        baseline_data_segmented['model_months'] = \
            [tuple([x]) for x in baseline_data_segmented.index.month]
    elif baseline_type in ['three_month', 'three_month_weighted']:
        unique_months = pd.Series(
                baseline_data.index
                .map(lambda x: x.month)
                .unique().values) \
                .map(lambda x: (x,))
        months = pd.DataFrame(unique_months, columns=['model_months'])

        def shoulder_months(month):
            if month == 1:
                return (12, 1, 2)
            elif month == 12:
                return (11, 12, 1)
            else:
                return (month - 1, month, month + 1)

        months.loc[:, 'baseline'] = months.model_months \
            .map(lambda x: shoulder_months(x[0]))
        for i, month in months.iterrows():
            this_df = baseline_data \
                    .loc[baseline_data.index.month.isin(month.baseline)]
            this_df.loc[:, 'model_months'] = \
                [month.model_months] * len(this_df.index)
            warnings.extend(get_hourly_coverage_warning(
                    this_df, month.baseline, month.model_months))

            this_df['weight'] = 1
            if baseline_type == 'three_month_weighted':
                this_df.loc[
                        [x[0] not in x[1] for x in
                         zip(this_df.index.month, this_df.model_months)],
                        'weight'] = 0.5
            baseline_data_segmented = baseline_data_segmented.append(
                    this_df, sort=False)
    elif baseline_type == 'single':
        baseline_data_segmented = baseline_data.copy()
        baseline_data_segmented['weight'] = 1
        baseline_data_segmented['model_months'] = \
            [range(1, 13) for j in range(len(baseline_data_segmented.index))]

    baseline_data_segmented = baseline_data_segmented.reset_index()
    warnings.extend(get_calendar_year_coverage_warning(
            baseline_data_segmented))
    return baseline_data_segmented, warnings


def get_feature_hour_of_week(data):
    warnings = []
    feature_hour_of_week = \
        data.apply(lambda x: (x.name.dayofweek) * 24 +
                   (x.name.hour+1), axis=1) \
            .reset_index() \
            .rename(columns={0: 'hour_of_week'})
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
