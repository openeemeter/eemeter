import numpy as np
import pandas as pd
import pytest

from eemeter import (
    load_sample,
    merge_temperature_data,
    get_baseline_data,
    assign_baseline_periods
)


# E2E Test
@pytest.fixture
def utc_index():
    return pd.date_range('2011-01-01', freq='H', periods=365*24 + 1, tz='UTC')


@pytest.fixture
def temperature_data(utc_index):
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-hourly')
    return temperature_data


@pytest.fixture
def meter_data():
    meter_data, temperature_data, metadata = \
        load_sample('il-electricity-cdd-hdd-hourly')
    return meter_data


@pytest.fixture
def baseline_data(meter_data, temperature_data):
    merged_data = merge_temperature_data(meter_data, temperature_data)
    baseline_data, warnings = get_baseline_data(
            data=merged_data, end=merged_data.index[-1], max_days=365)
    return baseline_data


def test_e2e(
        meter_data, temperature_data):
#    meter_data = meter_data()
#    temperature_data = temperature_data(utc_index())
    # Merge meter and temperature data
    merged_data = merge_temperature_data(meter_data, temperature_data)
    assert len(merged_data.index) == len(meter_data.index)
    assert np.mean(merged_data.temperature_mean) == np.mean(temperature_data)

    # Filter to 365 day baseline
    baseline_data, warnings = get_baseline_data(
            data=merged_data, end=merged_data.index[-1], max_days=365)
    assert baseline_data.shape == (8761, 4)

    # Calculate occupancy lookup table
    baseline_data, warnings = assign_baseline_periods(
            baseline_data, baseline_type='three_month_weighted')
    assert all(column in baseline_data.columns
               for column in ['weight', 'model_months'])
    # Validate temperature bin endpoints and determine temperature bins

    # Generate design matrix for weighted 3-month baseline

    # Fit consumption model

    # Use fitted model to predict counterfactual in reporting period


# Unit tests

def test_assign_baseline_periods_wrong_baseline_type(baseline_data):
    with pytest.raises(ValueError) as error:
        baseline_data_segmented, warnings = assign_baseline_periods(
            baseline_data, baseline_type='unknown')
    assert 'Invalid baseline type' in str(error)


def test_assign_baseline_periods_missing_temperature_data(baseline_data):
    baseline_data = baseline_data.drop('temperature_mean', axis=1)
    with pytest.raises(ValueError) as error:
        baseline_data_segmented, warnings = assign_baseline_periods(
                baseline_data, baseline_type='three_month_weighted')
    assert 'Data does not include columns' in str(error)


def test_assign_baseline_periods_one_month(baseline_data):
    baseline_data_segmented, warnings = assign_baseline_periods(
            baseline_data, baseline_type='one_month')

    unique_models = [
            list(x) for x in
            set(tuple(x) for x in baseline_data_segmented.model_months)]
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 12
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_months'])
    assert baseline_data_segmented.shape == (8761, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)])
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)] == 1)


def test_assign_baseline_periods_three_month(baseline_data):
    baseline_data_segmented, warnings = assign_baseline_periods(
            baseline_data, baseline_type='three_month')

    unique_models = [
            list(x) for x in
            set(tuple(x) for x in baseline_data_segmented.model_months)]
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 12
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_months'])
    assert baseline_data_segmented.shape == (8761*3, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)])
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)] == 1)
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           not in x.model_months, axis=1)] == 1)


def test_assign_baseline_periods_three_month_weighted(baseline_data):
    baseline_data_segmented, warnings = assign_baseline_periods(
            baseline_data, baseline_type='three_month_weighted')

    unique_models = [
            list(x) for x in
            set(tuple(x) for x in baseline_data_segmented.model_months)]
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 12
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_months'])
    assert baseline_data_segmented.shape == (8761*3, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)])
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)] == 1)
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           not in x.model_months, axis=1)] != 1)


def test_assign_baseline_periods_single(baseline_data):
    baseline_data_segmented, warnings = assign_baseline_periods(
            baseline_data, baseline_type='single')

    unique_models = [
            list(x) for x in
            set(tuple(x) for x in baseline_data_segmented.model_months)]
    captured_months = [element for sublist in unique_models
                       for element in sublist]
    assert len(warnings) == 0
    assert all(month in captured_months for month in range(1, 13))
    assert len(unique_models) == 1
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_months'])
    assert baseline_data_segmented.shape == (8761, 6)
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)])
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)] == 1)


def test_assign_baseline_periods_three_months_wtd_truncated(baseline_data):
    baseline_data, warnings = get_baseline_data(
            data=baseline_data, end=baseline_data.index[-1], max_days=180)
    baseline_data_segmented, warnings = assign_baseline_periods(
            baseline_data, baseline_type='three_month_weighted')
    unique_models = [
            list(x) for x in
            set(tuple(x) for x in baseline_data_segmented.model_months)]
    assert len(warnings) == 7
    assert len(unique_models) == 7
    assert all(column in baseline_data_segmented.columns
               for column in ['meter_value', 'temperature_mean',
                              'weight', 'model_months'])
    assert np.sum(baseline_data.meter_value) == \
        np.sum(baseline_data_segmented.meter_value
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)])
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           in x.model_months, axis=1)] == 1)
    assert all(baseline_data_segmented.weight
               .loc[baseline_data_segmented
                    .apply(lambda x: x.name.month
                           not in x.model_months, axis=1)] != 1)
    assert warnings[-1].qualified_name == (
        'eemeter.caltrack_hourly'
        '.incomplete_calendar_year_coverage'
    )
    assert warnings[-1].description == (
        'Data does not cover full calendar year. '
        '5 Missing months: [3, 4, 5, 6, 7]'
    )
    assert warnings[-1].data == {'num_missing_months': 5,
                   'missing_months': [3, 4, 5, 6, 7]}

    
