import numpy as np
import pandas as pd
import statsmodels
import pytest

import eemeter

from eemeter import (
    load_sample,
    merge_temperature_data,
    get_baseline_data,
    segment_timeseries,
    get_feature_hour_of_week,
    get_feature_occupancy,
    get_feature_binned_temperatures,
    get_design_matrix,
    caltrack_hourly_method,
    caltrack_hourly_predict,
)

from eemeter.api import (
    HourlyModel,
    ModelFit,
)


@pytest.fixture
def merged_data():
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-hourly')
# TODO: Look into merge_temperature for hourly data
#    merged_data = merge_temperature_data(meter_data, temperature_data)
    merged_data = pd.DataFrame(meter_data) \
        .merge(pd.DataFrame(temperature_data),
               left_index=True, right_index=True) \
        .rename(columns={'value': 'meter_value',
                         'tempF': 'temperature_mean'})
    merged_data['n_days_dropped'] = 0
    merged_data['n_days_kept'] = 0
    return merged_data


@pytest.fixture
def baseline_data(merged_data):
    baseline_data, warnings = get_baseline_data(
            data=merged_data, end=merged_data.index[-1], max_days=365)
    return baseline_data


# E2E Test
def test_e2e(
        merged_data):
    e2e_warnings = []

    # Filter to 365 day baseline
    baseline_data, warnings = get_baseline_data(
            data=merged_data, end=merged_data.index[-1], max_days=365)
    e2e_warnings.extend(warnings)
    assert baseline_data.shape == (8761, 4)

    # Calculate baseline period segmentation
    baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='three_month_weighted')
    e2e_warnings.extend(warnings)
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value',
                              'temperature_mean', 'weight', 'model_id'])

    # Get hour of week feature
    feature_hour_of_week, hour_parameters, warnings = \
        get_feature_hour_of_week(baseline_data_segmented)
    e2e_warnings.extend(warnings)
    assert 'hour_of_week' in feature_hour_of_week.columns
    assert feature_hour_of_week.shape == \
        (len(baseline_data_segmented.index), 2)

    # Get occupancy feature
    feature_occupancy, occupancy_parameters, warnings = \
        get_feature_occupancy(baseline_data_segmented)
    e2e_warnings.extend(warnings)
    assert all(column in feature_occupancy.columns
               for column in ['model_id', 'occupancy'])
    assert feature_occupancy.shape == (len(baseline_data_segmented.index), 2)
    assert occupancy_parameters['occupancy_lookup'].shape == (168 * 12, 3)
    # Validate temperature bin endpoints and determine temperature bins

    # Generate design matrix for weighted 3-month baseline
    design_matrix, preprocessors_fit, warnings = \
        get_design_matrix(
                baseline_data_segmented,
                functions={
                        'get_feature_hour_of_week': {
                                'function': get_feature_hour_of_week,
                                'kwargs': {}
                             },
                        'get_feature_occupancy': {
                                'function': get_feature_occupancy,
                                'kwargs': {'occupancy_lookup':
                                           occupancy_parameters[
                                                    'occupancy_lookup']}
                                    }
                                 })
    e2e_warnings.extend(warnings)

    assert design_matrix.shape == (len(baseline_data_segmented.index), 8)
    assert all(column in design_matrix.columns
               for column in ['model_id', 'occupancy', 'hour_of_week'])
    assert all(key in preprocessors_fit.keys()
               for key in ['get_feature_occupancy',
                           'get_feature_hour_of_week'])
    # Fit consumption model
    preprocessors = {
            'segment_timeseries': {
                    'function': segment_timeseries,
                    'kwargs': {'segment_type': 'three_month_weighted'}},
            'get_feature_hour_of_week': {
                    'function': get_feature_hour_of_week,
                    'kwargs': {}
                    },
            'get_feature_occupancy': {
                    'function': get_feature_occupancy,
                    'kwargs': {'threshold': 0.6}
                        },
            'get_feature_binned_temperatures': {
                    'function': get_feature_binned_temperatures,
                    'kwargs': {}}
                        }
    formula = ('meter_value ~ C(hour_of_week) - 1 + '
               'bin_0:occupancy + '
               'bin_1:occupancy + bin_2:occupancy + '
               'bin_3:occupancy + bin_4:occupancy + '
               'bin_5:occupancy + bin_6:occupancy')
    model_fit = caltrack_hourly_method(
            baseline_data, formula, preprocessors)
    e2e_warnings.extend(model_fit.warnings)
    assert isinstance(model_fit, ModelFit)
    assert model_fit.status == 'SUCCESS'
    assert model_fit.method_name == 'caltrack_hourly_method'
    assert model_fit.settings == {'preprocessors': preprocessors,
                                  'formula': formula}
    assert isinstance(model_fit.model, HourlyModel)
    assert len(model_fit.model.model_object) == 12
    assert isinstance(list(model_fit.model.model_object.values())[0],
                      statsmodels.regression.linear_model.WLS)
    assert isinstance(model_fit.model.model_params, pd.DataFrame)
    assert model_fit.model.model_params.shape == (12, 1 + 168 + 7)
    assert model_fit.model.__repr__() == \
        ("HourlyModel(segment_type='three_month_weighted', "
         "formula='{}', "
         "status='SUCCESS')").format(formula)
    # Use fitted model to predict counterfactual in reporting period
    model_prediction = model_fit.model.predict(baseline_data)
    e2e_warnings.extend(model_prediction.warnings)
    assert len(model_prediction.result.index) == len(baseline_data.index)
    #assert results.shape >= (len(baseline_data.index), 1)
    assert len(e2e_warnings) == 0


# Unit tests
def test_segment_timeseries_wrong_baseline_type(baseline_data):
    with pytest.raises(ValueError) as error:
        baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='unknown')
    assert 'Invalid segment type' in str(error)


def test_segment_timeseries_missing_temperature_data(baseline_data):
    baseline_data = baseline_data.drop('temperature_mean', axis=1)
    with pytest.raises(ValueError) as error:
        baseline_data_segmented, warnings = segment_timeseries(
                baseline_data, segment_type='three_month_weighted')
    assert 'Data does not include columns' in str(error)


def test_segment_timeseries_already_segmented(baseline_data):
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    with pytest.raises(ValueError) as error:
        baseline_data_segmented, warnings = segment_timeseries(
                baseline_data, segment_type='three_month_weighted')
    assert 'Data already contains column' in str(error)


def test_segment_timeseries_one_month(baseline_data):
    baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='one_month')

    unique_models = baseline_data_segmented.model_id.unique()
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 12
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_id'])
    assert baseline_data_segmented.shape == (8761, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]])
    assert all(baseline_data_segmented.weight
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] == 1)


def test_segment_timeseries_three_month(baseline_data):
    baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='three_month')

    unique_models = baseline_data_segmented.model_id.unique()
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 12
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_id'])
    assert baseline_data_segmented.shape == (8761*3, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]])
    assert all(baseline_data_segmented.weight
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] == 1)
    assert all(baseline_data_segmented.weight
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] == 1)


def test_segment_timeseries_three_month_weighted(baseline_data):
    baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='three_month_weighted')

    unique_models = baseline_data_segmented.model_id.unique()
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 12
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_id'])
    assert baseline_data_segmented.shape == (8761*3, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]])
    assert all(baseline_data_segmented.weight
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] == 1)
    assert all(baseline_data_segmented.weight
               .loc[[x[0] not in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] != 1)


def test_segment_timeseries_single(baseline_data):
    baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='single')

    unique_models = baseline_data_segmented.model_id.unique()
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 1
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_id'])
    assert baseline_data_segmented.shape == (8761, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]])
    assert all(baseline_data_segmented.weight
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] == 1)


def test_segment_timeseries_three_months_wtd_truncated(merged_data):
    baseline_data, warnings = get_baseline_data(
            data=merged_data, end=merged_data.index[-1], max_days=180)
    baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='three_month_weighted')
    unique_models = baseline_data_segmented.model_id.unique()
    assert len(warnings) == 7
    assert len(unique_models) == 7
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_id'])
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]])
    assert all(baseline_data_segmented.weight
               .loc[[x[0] not in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] != 1)
    assert warnings[-1].qualified_name == (
        'eemeter.caltrack_hourly'
        '.incomplete_calendar_year_coverage'
    )
    assert warnings[-1].description == (
        'Data does not cover full calendar year. '
        '5 Missing monthly models: [3, 4, 5, 6, 7]'
    )
    assert warnings[-1].data == {'num_missing_months': 5,
                                 'missing_months': [3, 4, 5, 6, 7]}


def test_segment_timeseries_three_months_wtd_insufficient(merged_data):
    baseline_data, warnings = get_baseline_data(
            data=merged_data, end=merged_data.index[-1], max_days=360)
    baseline_data_segmented, warnings = segment_timeseries(
            baseline_data, segment_type='three_month_weighted')
    unique_models = baseline_data_segmented.model_id.unique()
    ndays = baseline_data.index[0].days_in_month
    assert len(warnings) == 3
    assert len(unique_models) == 12
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_id'])
    assert round(np.sum(baseline_data.meter_value), 4) == \
        round(np.sum(baseline_data_segmented.meter_value
                     .loc[[x[0] in x[1] for x in
                           zip(baseline_data_segmented
                               .index.month,
                               baseline_data_segmented.model_id)]]), 4)
    assert all(baseline_data_segmented.weight
               .loc[[x[0] in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] == 1)
    assert all(baseline_data_segmented.weight
               .loc[[x[0] not in x[1] for x in
                     zip(baseline_data_segmented
                         .index.month,
                         baseline_data_segmented.model_id)]] != 1)
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.insufficient_hourly_coverage'
    )
    assert ('Data for this model does not meet the minimum hourly '
            'sufficiency criteria. Month 2') in warnings[0].description
    assert round(warnings[0].data['hourly_coverage'], 4) == \
        round(((ndays - 5) * 24.0 + 1)/(ndays * 24), 4)


def test_feature_hour_of_week(baseline_data):
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1
    feature_hour_of_week, parameters, warnings = get_feature_hour_of_week(
            baseline_data)
    assert len(warnings) == 0
    assert all(column in feature_hour_of_week.columns
               for column in ['hour_of_week', 'model_id'])
    assert feature_hour_of_week.shape == (len(baseline_data.index), 2)
    assert all(x in feature_hour_of_week.hour_of_week.unique()
               for x in range(1, 169))
    assert feature_hour_of_week.hour_of_week.dtypes == 'category'


def test_feature_hour_of_week_incomplete_week(merged_data):
    five_day_index = pd.date_range('2017-01-04', freq='H',
                                   periods=5*24, tz='UTC',
                                   name='start')
    baseline_data = pd.DataFrame({'meter_value': [1 for i in range(0, 120)]}) \
        .set_index(five_day_index)
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1
    feature_hour_of_week, parameters, warnings = get_feature_hour_of_week(
            baseline_data)
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.missing_hours_of_week'
    )
    assert ('Data does not include all hours of week.') \
        in warnings[0].description
    assert warnings[0].data['num_missing_hours'] == 24 * 2


def test_feature_occupancy_unsegmented(baseline_data):
    feature_occupancy, parameters, warnings = \
        get_feature_occupancy(baseline_data, threshold=0.5)

    assert feature_occupancy.shape == (len(baseline_data.index), 2)
    assert parameters['occupancy_lookup'].shape == (168, 3)
    assert sum(parameters['occupancy_lookup'].occupancy) == 4
    assert len(warnings) == 2
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.missing_model_id'
    )
    assert warnings[1].qualified_name == (
        'eemeter.caltrack_hourly'
        '.missing_weight_column'
    )
    assert all(column not in warnings[0].data['dataframe_columns']
               for column in ['model_id', 'weight'])


def test_feature_occupancy_failed_model(baseline_data):
    baseline_data = baseline_data.drop('meter_value', axis=1)
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1
    feature_occupancy, parameters, warnings = \
        get_feature_occupancy(baseline_data)
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.failed_occupancy_model'
    )
    assert 'Error encountered in statsmodels.formula.api.wls' \
        in warnings[0].description
    assert warnings[0].data['traceback'] is not None


def test_feature_binned_temperatures_unsegmented(baseline_data):
    feature_temperature, parameters, warnings = \
        get_feature_binned_temperatures(baseline_data)

    assert feature_temperature.shape == (len(baseline_data.index), 8)
    assert parameters['temperature_bins'].shape == (1, 2)
    assert len(warnings) == 2
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.missing_model_id'
    )
    assert warnings[1].qualified_name == (
        'eemeter.caltrack_hourly'
        '.missing_weight_column'
    )
    assert all(column not in warnings[0].data['dataframe_columns']
               for column in ['model_id', 'weight'])


def test_feature_binned_temperatures_wrong_bin_arg(baseline_data):
    bins = pd.DataFrame({
            'model_id': [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)],
            'valid_bins': [[222, 333]]})
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1
    feature_temperature, parameters, warnings = \
        get_feature_binned_temperatures(
                baseline_data, mode='predict',
                temperature_bins=bins)
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.temperature_bins_failed_read'
    )
    assert 'Provided temperature bins do not match required format.' \
        in warnings[0].description


def test_feature_binned_temperatures_incomplete_range(baseline_data):
    data = pd.DataFrame({
            'temperature_mean': [i for i in range(70)]},
        index=pd.date_range('2017-01-04', freq='H', periods=70, tz='UTC',
                            name='start'))
    feature_temperature, parameters, warnings = \
        get_feature_binned_temperatures(
                data,
                default_bins=[30, 45, 55, 65, 75, 90],
                min_temperature_count=5)
    assert all(column in feature_temperature.columns
               for column in ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
                              'bin_5', 'bin_6'])
    assert any(feature_temperature.bin_0 != 0.0)
    assert any(feature_temperature.bin_1 != 0.0)
    assert any(feature_temperature.bin_2 != 0.0)
    assert any(feature_temperature.bin_3 != 0.0)
    assert all(feature_temperature.bin_4 == 0.0)
    assert all(feature_temperature.bin_5 == 0.0)
    assert all(feature_temperature.bin_6 == 0.0)


def test_feature_assign_temperatures_bins_no_valid_bins(baseline_data):
    feature_temperature = eemeter.caltrack_hourly.assign_temperature_bins(
            baseline_data, [])
    assert all(feature_temperature.bin_0.iloc[0:-1] ==
               baseline_data.temperature_mean.iloc[0:-1])


def test_feature_assign_temperatures_bins_off_limit():
    data = pd.DataFrame({
            'temperature_mean': [i for i in range(24)]},
        index=pd.date_range('2017-01-04', freq='H', periods=24, tz='UTC',
                            name='start'))
    feature_temperature = eemeter.caltrack_hourly.assign_temperature_bins(
            data, [60, 80]).dropna()
    assert feature_temperature.shape == (len(data.index), 3)
    assert any(feature_temperature.bin_0 != 0.0)
    assert all(feature_temperature.bin_1 == 0.0)
    assert all(feature_temperature.bin_2 == 0.0)


def test_validate_temperature_bins_single_temperature(baseline_data):
    data = pd.DataFrame({
            'temperature_mean': [55] * 24},
        index=pd.date_range('2017-01-04', freq='H', periods=24, tz='UTC',
                            name='start'))
    temperature_summary, bin_endpoints_valid = \
        eemeter.caltrack_hourly.validate_temperature_bins(data, [30, 60, 90])
    assert temperature_summary.shape == (4, 2)
    assert bin_endpoints_valid == []


def test_get_design_matrix_different_length_index(baseline_data):
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1

    def get_ones(data, n):
        n_day_index = pd.date_range('2017-01-04', freq='H',
                                    periods=n*24, tz='UTC',
                                    name='start')
        weird_feature = pd.DataFrame(
                {'meter_value': [1] * (n * 24)}) \
            .set_index(n_day_index)
        return weird_feature, {}, []
    design_matrix, feature_parameters, warnings = \
        get_design_matrix(
                baseline_data,
                functions={
                        'get_ones': {
                                'function': get_ones,
                                'kwargs': {'n': 5}
                         },
                        })
    assert len(design_matrix.index) == 0
    assert len(feature_parameters) == 0
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.design_matrix_unmatched_index'
    )
    assert 'Function returned a feature whose index does not match the data' \
        in warnings[0].description
    assert warnings[0].data['function'] == 'get_ones'


def test_get_design_matrix_unmatched_index(baseline_data):
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1

    def get_ones(data):
        n_day_index = pd.date_range('2017-01-04', freq='H',
                                    periods=len(data.index), tz='UTC',
                                    name='start')
        weird_feature = pd.DataFrame(
                {'meter_value': [1] * len(data.index)}) \
            .set_index(n_day_index)
        return weird_feature, {}, []
    design_matrix, feature_parameters, warnings = \
        get_design_matrix(
                baseline_data,
                functions={
                        'get_ones': {
                                'function': get_ones,
                                'kwargs': {}
                         },
                        })
    assert len(design_matrix.index) == 0
    assert len(feature_parameters) == 0
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.design_matrix_unmatched_index'
    )
    assert 'Function returned a feature whose index does not match the data' \
        in warnings[0].description
    assert warnings[0].data['function'] == 'get_ones'


def test_get_design_matrix_unsegmented(baseline_data):
    design_matrix, preprocessors_fit, warnings = \
        get_design_matrix(
                baseline_data,
                functions={
                        'get_feature_hour_of_week': {
                                'function': get_feature_hour_of_week,
                                'kwargs': {}
                         },
                        })

    assert len(design_matrix.index) == 8761
    assert len(preprocessors_fit['get_feature_hour_of_week']['kwargs']) == 0
    assert len(warnings) == 2
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.missing_model_id'
    )
    assert warnings[1].qualified_name == (
        'eemeter.caltrack_hourly'
        '.missing_weight_column'
    )
    assert all(column not in warnings[0].data['dataframe_columns']
               for column in ['model_id', 'weight'])


def test_get_design_matrix_with_segmentation_function(baseline_data):
    design_matrix, preprocessors_fit, warnings = \
        get_design_matrix(
                baseline_data,
                functions={
                        'segment_timeseries': {
                                'function': segment_timeseries,
                                'kwargs': {'segment_type': 'one_month'}
                         },
                        'get_feature_hour_of_week': {
                                'function': get_feature_hour_of_week,
                                'kwargs': {}
                         },
                        })

    assert len(design_matrix.index) == 8761
    assert len(preprocessors_fit['get_feature_hour_of_week']['kwargs']) == 0
    assert len(warnings) == 0


def test_get_design_matrix_with_segmentation_missing_arg(baseline_data):
    design_matrix, preprocessors_fit, warnings = \
        get_design_matrix(
                baseline_data,
                functions={
                        'segment_timeseries': {
                                'function': segment_timeseries,
                                'kwargs': {}
                         },
                        'get_feature_hour_of_week': {
                                'function': get_feature_hour_of_week,
                                'kwargs': {}
                         },
                        })

    assert len(design_matrix.index) == 0
    assert len(preprocessors_fit) == 0
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.design_matrix_wrong_kwargs'
    )
    assert 'Error: Missing or wrong keyword arguments' in \
        warnings[0].description
    assert warnings[0].data['function'] == 'segment_timeseries'
    assert warnings[0].data['kwargs'] == {}
    assert warnings[0].data['traceback'] is not None


def test_get_design_matrix_wrong_kwargs(baseline_data):
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1
    design_matrix, feature_parameters, warnings = \
        get_design_matrix(
                baseline_data,
                functions={
                        'get_feature_hour_of_week': {
                                'function': get_feature_hour_of_week,
                                'kwargs': {'wrong': 55}
                         },
                        })

    assert len(design_matrix.index) == 0
    assert len(feature_parameters) == 0
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.design_matrix_wrong_kwargs'
    )
    assert 'Error: Missing or wrong keyword arguments' in \
        warnings[0].description
    assert warnings[0].data['function'] == 'get_feature_hour_of_week'
    assert warnings[0].data['kwargs'] == {'wrong': 55}
    assert warnings[0].data['traceback'] is not None


def test_get_design_matrix_wrong_function_schema(baseline_data):
    baseline_data['model_id'] = [(1,)] * len(baseline_data.index)
    baseline_data['weight'] = 1
    design_matrix, feature_parameters, warnings = \
        get_design_matrix(
                baseline_data,
                functions={
                        'get_feature_hour_of_week': {
                                'function': 'get_feature_hour_of_week',
                                'kwargs': {}
                         },
                        })

    assert len(design_matrix.index) == 0
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.design_matrix_wrong_schema'
    )
    assert 'Wrong schema for function list. Expecting dict of dicts.' in \
        warnings[0].description
    assert warnings[0].data['traceback'] is not None


def test_caltrack_hourly_method_no_data():
    data = pd.DataFrame({
        'meter_value': [],
        'hour_of_week': [],
    })
    model_fit = caltrack_hourly_method(
            data, formula='meter_value ~ hour_of_week')
    assert model_fit.method_name == 'caltrack_hourly_method'
    assert model_fit.status == 'NO DATA'
    assert len(model_fit.warnings) == 1
    warning = model_fit.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_hourly.no_data'
    )
    assert warning.description == (
        'No data available. Cannot fit model.'
    )
    assert warning.data == {}


def test_caltrack_hourly_method_formula_does_not_match_data():
    data = pd.DataFrame({
        'meter_value': [1, 2, 1],
        'hour_of_week': [2, 3, 4],
    })
    model_fit = caltrack_hourly_method(
            data, formula='meter_value ~ hour_of_week + missing_feature')
    assert model_fit.method_name == 'caltrack_hourly_method'
    assert model_fit.status == 'NOT ATTEMPTED'
    assert len(model_fit.warnings) == 1
    warning = model_fit.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_hourly.missing_features'
    )
    assert warning.description == (
        'Data is missing features specified in formula.'
    )
    assert warning.data['formula'] == (
        'meter_value ~ hour_of_week + missing_feature'
    )
    assert sorted(warning.data['dataframe_columns']) == [
        'hour_of_week', 'meter_value', 'model_id', 'weight'
    ]


def test_caltrack_hourly_method_failed_model():
    data = pd.DataFrame({
        'meter_value': [1, 2, 1],
        'hour_of_week': [2, 3, 4],
        'weight': [np.inf, np.inf, np.inf]
    })
    model_fit = caltrack_hourly_method(
            data, formula='meter_value ~ hour_of_week')
    assert model_fit.method_name == 'caltrack_hourly_method'
    assert model_fit.status == 'FAILED'
    assert len(model_fit.warnings) == 1
    warning = model_fit.warnings[-1]
    assert warning.qualified_name == (
        'eemeter.caltrack_hourly.failed_consumption_model'
    )
    assert warning.description == (
        'Error encountered in statsmodels.formula.api.wls method '
        'for model id: (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)'
    )
    assert warning.data is not None


def test_caltrack_hourly_method_no_formula():
    data = pd.DataFrame({
        'meter_value': [1, 2, 1],
        'hour_of_week': [2, 3, 4],
        'weight': [1, 1, 1],
        'occupancy': [0, 1, 0],
        'bin_0': [22, 44, 66],
        'bin_1': [22, 44, 66],
        'bin_2': [22, 44, 66],
        'bin_3': [22, 44, 66],
        'bin_4': [22, 44, 66],
        'bin_5': [22, 44, 66],
        'bin_6': [22, 44, 66],
    })
    model_fit = caltrack_hourly_method(data)
    assert model_fit.method_name == 'caltrack_hourly_method'
    assert model_fit.status == 'SUCCESS'
    assert len(model_fit.warnings) == 1
    warning = model_fit.warnings[-1]
    assert warning.qualified_name == (
        'eemeter.caltrack_hourly.missing_model_id'
    )
    assert warning.description == (
        'Warning: Model ID is missing - this function will be run '
        'with all of the data in a single model'
    )
    assert warning.data is not None


def test_caltrack_hourly_predict_no_temperature():
    data = pd.DataFrame({
        'meter_value': [1, 2, 1, 6, 2]
    })
    time_index = pd.date_range('2017-01-04', freq='H',
                               periods=5, tz='UTC', name='start')
    data = data.set_index(time_index)
    with pytest.raises(ValueError) as error:
        results, design_matrix, warnings = caltrack_hourly_predict(
            None, None, None, None, data)
    assert 'Data does not include a temperature_mean column' in str(error)


def test_caltrack_hourly_predict_no_datetime_index():
    data = pd.DataFrame({
        'temperature_mean': [1, 2, 1, 6, 2]
    })
    with pytest.raises(TypeError) as error:
        results, design_matrix, warnings = caltrack_hourly_predict(
            None, None, None, None, data)
    assert 'Dataframe index is not a pandas DatetimeIndex' in str(error)


def test_caltrack_hourly_predict_design_matrix_missing_model_parameters():
    data = pd.DataFrame({
        'temperature_mean': [1, 2, 1, 6, 2],
        'hour_of_week': [1, 2, 3, 4, 5]
    })
    time_index = pd.date_range('2017-01-04', freq='H',
                               periods=5, tz='UTC', name='start')
    data = data.set_index(time_index)
    unique_models = [(1,)]
    formula = 'meter_value ~ C(hour_of_week) - 1'
    with pytest.raises(TypeError) as error:
        results, design_matrix, warnings = caltrack_hourly_predict(
                formula, None, unique_models, None, data)
    assert 'Model parameters haven\'t been provided' in str(error)


def test_caltrack_hourly_predict_design_matrix_missing_formula():
    data = pd.DataFrame({
        'temperature_mean': [1, 2, 1, 6, 2],
        'hour_of_week': [1, 2, 3, 4, 5]
    })
    time_index = pd.date_range('2017-01-04', freq='H',
                               periods=5, tz='UTC', name='start')
    data = data.set_index(time_index)
    unique_models = [(1,)]
    with pytest.raises(TypeError) as error:
        results, design_matrix, warnings = caltrack_hourly_predict(
                None, None, unique_models, None, data)
    assert 'Formula hasn\'t been provided' in str(error)


def test_caltrack_hourly_predict_design_matrix_missing_columns():
    data = pd.DataFrame({
        'temperature_mean': [1, 2, 1, 6, 2]
    })
    model_params = pd.DataFrame()
    time_index = pd.date_range('2017-01-04', freq='H',
                               periods=5, tz='UTC', name='start')
    data = data.set_index(time_index)
    unique_models = [(1,)]
    formula = 'meter_value ~ C(hour_of_week) - 1'
    results, design_matrix, warnings = caltrack_hourly_predict(
            formula, None, unique_models, model_params, data)
    assert warnings[0].qualified_name == \
        'eemeter.caltrack_hourly.missing_features'
    assert warnings[0].description == \
        'Data is missing features specified in formula.'
    assert warnings[0].data == {
            'dataframe_columns': ['temperature_mean', 'model_id', 'weight'],
            'formula': 'meter_value ~ C(hour_of_week) - 1'}
    assert all(column in design_matrix.columns
               for column in ['temperature_mean', 'model_id'])


def test_caltrack_hourly_predict_design_matrix_failure(baseline_data):
    data = pd.DataFrame({
        'temperature_mean': [1, 2, 1, 6, 2],
        'hour_of_week': [1, 2, 3, 4, 5]
    })
    time_index = pd.date_range('2017-01-04', freq='H',
                               periods=5, tz='UTC', name='start')
    data = data.set_index(time_index)
    unique_models = [(1,)]
    model_params = pd.DataFrame()
    formula = 'meter_value ~ C(hour_of_week) - 1'
    preprocessors = {
                    'get_feature_hour_of_week': {
                            'function': 'wrong',
                            'kwargs': {}
                     },
                    }
    results, design_matrix, warnings = caltrack_hourly_predict(
            formula, preprocessors, unique_models, model_params, data)

    assert len(design_matrix.index) == 0
    assert len(results.index) == 0
    assert len(warnings) == 1
    assert warnings[0].qualified_name == (
        'eemeter.caltrack_hourly'
        '.design_matrix_wrong_schema'
    )
    assert 'Wrong schema for function list. Expecting dict of dicts.' in \
        warnings[0].description
    assert warnings[0].data['traceback'] is not None


def test_caltrack_hourly_error_propagation():
    pass
