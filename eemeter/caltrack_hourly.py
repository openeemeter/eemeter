import numpy as np
import pandas as pd
from .api import (
    EEMeterWarning,
    ModelFit,
    HourlyModel,
)
import statsmodels.formula.api as smf
from patsy import ModelDesc, dmatrix
import traceback
import warnings as ws
from schema import Schema, SchemaError
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


def segment_timeseries(data, segment_type):

    data_segmented = pd.DataFrame()
    warnings = []
    valid_segment_types = ['one_month',
                           'three_month',
                           'three_month_weighted',
                           'single', ]
    if segment_type not in valid_segment_types:
        raise ValueError('Invalid segment type: %s' % (segment_type))

    for column in ['meter_value', 'temperature_mean']:
        if column not in data.columns:
            raise ValueError('Data does not include columns: {}'
                             .format(column))

    if segment_type == 'one_month':
        data_segmented = data.copy()
        data_segmented['weight'] = 1
        data_segmented['model_id'] = \
            [tuple([x]) for x in data_segmented.index.month]
    elif segment_type in ['three_month', 'three_month_weighted']:
        unique_months = pd.Series(
                data.index
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
            this_df = data \
                    .loc[data.index.month.isin(month.baseline)]
            this_df.loc[:, 'model_id'] = \
                [month.model_id] * len(this_df.index)
            warnings.extend(get_hourly_coverage_warning(
                    this_df, month.baseline, month.model_id))

            this_df['weight'] = 1
            if segment_type == 'three_month_weighted':
                this_df.loc[
                        [x[0] not in x[1] for x in
                         zip(this_df.index.month, this_df.model_id)],
                        'weight'] = 0.5
            data_segmented = data_segmented.append(
                    this_df, sort=False)
    elif segment_type == 'single':
        data_segmented = data.copy()
        data_segmented['weight'] = 1
        data_segmented['model_id'] = \
            [tuple(range(1, 13))] * len(data_segmented.index)

    warnings.extend(get_calendar_year_coverage_warning(
            data_segmented))
    return data_segmented, warnings


def get_missing_hours_of_week_warning(unique_hours):
    warning = []
    missing_hours = [hour for hour in range(1, 169)
                     if hour not in unique_hours]
    if sorted(unique_hours) != \
            [x for x in range(1, 169)]:
                warning = [EEMeterWarning(
                        qualified_name=('eemeter.caltrack_hourly.'
                                        'missing_hours_of_week'),
                        description=(
                                'Data does not include all hours of week. '
                                'Missing hours of week: {}'
                                .format(missing_hours)
                                ),
                        data={'num_missing_hours': 168 - len(unique_hours),
                              'missing_hours': missing_hours}
                        )]
    return warning


def get_feature_hour_of_week(data):
    hour_warnings = []

    data_verified, warnings = handle_unsegmented_timeseries(data)
    hour_warnings.extend(warnings)

    feature_hour_of_week = \
        pd.DataFrame({
                'hour_of_week': data.index.dayofweek * 24
                + (data.index.hour + 1),
                'model_id': data.model_id},
            index=data.index)
    feature_hour_of_week["hour_of_week"] = \
        feature_hour_of_week["hour_of_week"].astype('category')

    hour_warnings.extend(get_missing_hours_of_week_warning(
            feature_hour_of_week.hour_of_week.unique()))

    parameters = {}
    return feature_hour_of_week, parameters, hour_warnings


def get_fit_failed_model_warning(model_id, model_type):

    warning = [EEMeterWarning(
        qualified_name='eemeter.caltrack_hourly.failed_' + model_type,
        description=(
            'Error encountered in statsmodels.formula.api.wls method '
            'for model id: {}'.format(model_id)
        ),
        data={'model_id': model_id,
              'model_type': model_type,
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
                get_fit_failed_model_warning(data.model_id[0],
                                             'occupancy_model'))
        return None, pd.DataFrame(), warnings

    model_data = model_data.merge(
            pd.DataFrame({'residuals': model_occupancy.fit().resid}),
            left_index=True, right_index=True)

    # TODO: replace with design matrix function
    feature_hour_of_week, parameters, warnings = get_feature_hour_of_week(data)
    model_data = model_data.merge(feature_hour_of_week,
                                  left_index=True, right_index=True)

    occupancy_lookup = pd.DataFrame({
            'occupancy': model_data.groupby(['hour_of_week'])
            .apply(lambda x: ishighusage(x, threshold))}) \
        .reset_index()

    return model_occupancy, occupancy_lookup, warnings


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


def get_feature_occupancy(data, mode='fit',
                          threshold=0.65, occupancy_lookup=None,
                          **kwargs):
    occupancy_warnings = []
    occupancy_models = {}

    data_verified, warnings = handle_unsegmented_timeseries(data)
    occupancy_warnings.extend(warnings)

    if mode == 'fit':
        occupancy_lookup = pd.DataFrame()
        unique_models = data_verified.model_id.unique()
        for model_id in unique_models:
            this_data = data_verified.loc[data_verified.model_id == model_id]
            this_model, this_occupancy_lookup, this_warnings = \
                get_single_feature_occupancy(this_data, threshold)
            this_occupancy_lookup['model_id'] = [model_id] * \
                len(this_occupancy_lookup.index)

            occupancy_lookup = occupancy_lookup.append(this_occupancy_lookup,
                                                       sort=False)
            occupancy_warnings.extend(this_warnings)
            occupancy_models[model_id] = this_model

    if len(occupancy_lookup.index) == 0:
        return pd.DataFrame(), {}, occupancy_warnings

    feature_hour_of_week, parameters, warnings = \
        get_feature_hour_of_week(data_verified)

    feature_occupancy = data_verified.reset_index() \
        .merge(feature_hour_of_week,
               left_on=['start', 'model_id'],
               right_on=['start', 'model_id']) \
        .merge(occupancy_lookup,
               left_on=['model_id', 'hour_of_week'],
               right_on=['model_id', 'hour_of_week']) \
        .loc[:, ['start', 'model_id', 'occupancy']] \
        .set_index('start')
    occupancy_parameters = {
            'mode': 'predict',
            'occupancy_models': occupancy_models,
            'threshold': threshold,
            'occupancy_lookup': occupancy_lookup}
    return feature_occupancy, occupancy_parameters, occupancy_warnings


def get_design_matrix_unmatched_index_warning(function):
    warning = [EEMeterWarning(
        qualified_name='eemeter.caltrack_hourly.design_matrix_unmatched_index',
        description=(
            'Error: Function returned a feature whose index does not match '
            'the data. Function name: {}'.format(function['function'].__name__)
        ),
        data={'function': function['function'].__name__}
    )]
    return warning


def get_design_matrix_wrong_kwargs_warning(function):
    warning = [EEMeterWarning(
        qualified_name='eemeter.caltrack_hourly.design_matrix_wrong_kwargs',
        description=(
            'Error: Missing or wrong keyword arguments passed in function '
            'dict. Function name: {}'.format(function['function'].__name__)
        ),
        data={'function': function['function'].__name__,
              'kwargs': function['kwargs']}
    )]
    return warning


def handle_unsegmented_timeseries(data):
    data_verified = data.copy()
    warnings = []
    if 'model_id' not in data_verified.columns:
        warnings.extend(
                get_missing_model_id_warning(data_verified.columns))
        data_verified['model_id'] = \
            [tuple(range(1, 13))] * len(data_verified.index)
    if 'weight' not in data_verified.columns:
        warnings.extend(
                get_missing_weight_column_warning(data_verified.columns))
        data_verified['weight'] = 1
    return data_verified, warnings


def has_call_attribute(fn):
    return hasattr(fn, '__call__')


def get_invalid_function_dict_warning(function_dict):
    schema = Schema({str:
                    {'function': has_call_attribute,
                     'kwargs': dict}})
    try:
        schema.validate(function_dict)
    except SchemaError as e:
        return [EEMeterWarning(
            qualified_name=(
                'eemeter.caltrack_hourly.design_matrix_wrong_schema'
                ),
            description=(
                'Wrong schema for function list. Expecting dict of dicts.'
                ),
            data={'preprocessors': function_dict,
                  'traceback': traceback.format_exc()}
        )]
    return []


def get_design_matrix(data, functions):
    preprocessors_fit = {}
    design_matrix_warnings = []
    feature_functions = functions.copy()

    design_matrix_warnings.extend(
            get_invalid_function_dict_warning(feature_functions))
    if len(design_matrix_warnings) > 0:
        return pd.DataFrame(), dict(), design_matrix_warnings

    segmented = False
    for name, function in feature_functions.items():
        if function['function'].__name__ == 'segment_timeseries':
            try:
                data_verified, warnings = \
                    segment_timeseries(data, **function['kwargs'])
            except TypeError:
                design_matrix_warnings.extend(
                        get_design_matrix_wrong_kwargs_warning(function))
                return pd.DataFrame(), dict(), design_matrix_warnings
            segmented = True
            del(feature_functions[name])
            break

    if not segmented:
        data_verified, warnings = handle_unsegmented_timeseries(data)

    design_matrix_warnings.extend(warnings)

    design_matrix = data_verified.copy()
    for name, function in feature_functions.items():
        try:
            this_feature, this_parameters, this_warnings = \
                function['function'](data_verified, **function['kwargs'])
        except TypeError:
            design_matrix_warnings.extend(
                    get_design_matrix_wrong_kwargs_warning(function))
            return pd.DataFrame(), dict(), design_matrix_warnings

        if (len(this_feature.index) != len(data_verified.index)):
            design_matrix_warnings.extend(
                get_design_matrix_unmatched_index_warning(function))
            return pd.DataFrame(), dict(), design_matrix_warnings
        if (any(this_feature.sort_index().index !=
                data_verified.sort_index().index)):
            design_matrix_warnings.extend(
                get_design_matrix_unmatched_index_warning(function))
            return pd.DataFrame(), dict(), design_matrix_warnings

        design_matrix = design_matrix.merge(
                this_feature,
                left_on=['start', 'model_id'],
                right_on=['start', 'model_id'])
        preprocessors_fit.update(
                {name: {
                    'function': function['function'],
                    'kwargs': this_parameters
                    }})

    return design_matrix, preprocessors_fit, design_matrix_warnings


def get_single_model(data, formula):
    warnings = []

    try:
        with ws.catch_warnings(record=True) as sm_warnings:
            model_consumption = smf.wls(
                    formula=formula,
                    data=data,
                    weights=data.weight)
            if len(sm_warnings) > 0:
                raise RuntimeError([w.message for w in sm_warnings])
    except Exception as e:
        warnings.extend(
                get_fit_failed_model_warning(data.model_id[0],
                                             'consumption_model'))
        return None, pd.DataFrame(), warnings

    model_parameters = pd.DataFrame(model_consumption.fit().params).transpose()

    return model_consumption, model_parameters, warnings


def get_terms_in_formula(formula):
    model_desc = ModelDesc.from_formula(formula)
    term_list = []
    for side in [model_desc.lhs_termlist, model_desc.rhs_termlist]:
        for term in side:
            for factor in term.factors:
                term_list.extend([factor.name()])
    return pd.Series(term_list).str.replace('^C\(|\)', '').tolist()


def caltrack_hourly_method(data, formula=None, preprocessors=None):

    method_warnings = []
    if data.empty:
        return ModelFit(
            status='NO DATA',
            method_name='caltrack_hourly_method',
            warnings=[EEMeterWarning(
                qualified_name='eemeter.caltrack_hourly.no_data',
                description=(
                    'No data available. Cannot fit model.'
                ),
                data={},
            )],
        )

    segment_type = 'unknown'
    if preprocessors is None:
        design_matrix, warnings = handle_unsegmented_timeseries(data)
        method_warnings.extend(warnings)
        preprocessors_fit = {}
    else:
        design_matrix, preprocessors_fit, design_matrix_warnings = \
            get_design_matrix(data, preprocessors)
        if len(design_matrix_warnings) > 0:
            return ModelFit(
                status='NOT ATTEMPTED',
                method_name='caltrack_hourly_method',
                warnings=design_matrix_warnings,
            )
        for name, preprocessor in preprocessors.items():
            if preprocessor['function'] == segment_timeseries:
                segment_type = preprocessor['kwargs']['segment_type']

    if formula is None:
        formula = '''meter_value ~ C(hour_of_week) - 1 '''  #+
#                bin_lt30:occupancy +
#                bin_30_45:occupancy + bin_45_55:occupancy +
#                bin_55_65:occupancy + bin_65_75:occupancy +
#                bin_75_90:occupancy + bin_30_45:occupancy +
#                bin_gt90:occupancy''' # default Caltrack formula

    term_list = get_terms_in_formula(formula)
    if any(term not in design_matrix.columns for term in term_list):
        return ModelFit(
            status='NOT ATTEMPTED',
            method_name='caltrack_hourly_method',
            warnings=get_missing_features_warning(
                    formula, design_matrix.columns.tolist())
            )

    model_params = pd.DataFrame()
    model_object = {}
    model_warnings = []
    unique_models = design_matrix.model_id.unique()
    for model_id in unique_models:
        this_data = design_matrix.loc[design_matrix.model_id == model_id]
        this_model, this_parameters, this_warnings = \
            get_single_model(this_data, formula)
        this_parameters['model_id'] = [model_id] * \
            len(this_parameters.index)

        model_params = model_params.append(this_parameters, sort=False)
        model_warnings.extend(this_warnings)
        model_object[model_id] = this_model

    if len(model_warnings) > 0:
        return ModelFit(
            status='FAILED',
            method_name='caltrack_hourly_method',
            warnings=model_warnings,
        )
    model = HourlyModel(
        formula=formula,
        segment_type=segment_type,
        status='SUCCESS',
        warnings=method_warnings,
        predict_func=caltrack_hourly_predict,
        plot_func=None,  # TODO:
        model_params=model_params,
        model_object=model_object,
        preprocessors_raw=preprocessors,
        preprocessors_fit=preprocessors_fit,
        unique_models=unique_models
    )

    return ModelFit(
        status='SUCCESS',
        method_name='caltrack_hourly_method',
        model=model,
        candidates=None,
        warnings=method_warnings,
        settings={
                'preprocessors': preprocessors,
                'formula': formula},
    )


def get_missing_features_warning(formula, columns):
    warning = [EEMeterWarning(
                qualified_name=(
                    'eemeter.caltrack_hourly.missing_features'
                ),
                description=(
                    'Data is missing features specified in formula.'
                ),
                data={'formula': formula,
                      'dataframe_columns': columns},
            )]
    return warning


def caltrack_hourly_predict(formula, preprocessors_fit,
                            unique_models, model_params,
                            data):
    predict_warnings = []
    if 'temperature_mean' not in data.columns:
        raise ValueError('Data does not include a temperature_mean column')
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError('Dataframe index is not a pandas DatetimeIndex')
    if not isinstance(formula, str):
        raise TypeError('Formula hasn\'t been provided')
    if not isinstance(model_params, pd.DataFrame):
        raise TypeError('Model parameters haven\'t been provided')
    data_verified = data.copy()
    data_verified['model_id'] = [model
                                 for month in data_verified.index.month
                                 for model in unique_models
                                 if month in model]
    data_verified['weight'] = 1

    if preprocessors_fit is None:
        design_matrix = data_verified
    else:
        design_matrix, _fit, design_matrix_warnings = \
            get_design_matrix(data_verified, preprocessors_fit)
        if len(design_matrix_warnings) > 0:
            return ModelFit(
                status='NOT ATTEMPTED',
                method_name='caltrack_hourly_method',
                warnings=design_matrix_warnings,
            )

    term_list = get_terms_in_formula(formula.split('~')[1])
    if any(term not in design_matrix.columns for term in term_list):
        predict_warnings.extend(
                get_missing_features_warning(
                        formula, design_matrix.columns.tolist()))
        return pd.DataFrame(), design_matrix, predict_warnings

    design_matrix_granular = dmatrix(
            formula.split('~')[1],
            design_matrix, return_type='dataframe')
    design_matrix_granular['model_id'] = [
            model
            for month in design_matrix_granular.index.month
            for model in unique_models
            if month in model]

    results = pd.DataFrame()
    for model_id in unique_models:
        this_parameters = model_params \
            .loc[model_params.model_id == model_id] \
            .drop('model_id', axis=1)
        this_data = design_matrix_granular \
            .loc[design_matrix_granular.model_id == model_id] \
            .drop('model_id', axis=1)
        this_result = this_data.dot(this_parameters.transpose())
        results = results.append(this_result, sort=False)
    results = results.rename(columns={0: 'counterfactual'})

    return results, design_matrix, predict_warnings
