import json

import numpy as np
import pandas as pd
import pytest

from eemeter import (
    CandidateModel,
    caltrack_method,
    caltrack_sufficiency_criteria,
    caltrack_metered_savings,
    caltrack_modeled_savings,
    get_baseline_data,
    merge_temperature_data,
)
from eemeter.caltrack import (
    get_intercept_only_candidate_models,
    get_too_few_non_zero_degree_day_warning,
    get_total_degree_day_too_low_warning,
    get_parameter_negative_warning,
    get_parameter_p_value_too_high_warning,
    get_cdd_only_candidate_models,
    get_hdd_only_candidate_models,
    get_cdd_hdd_candidate_models,
    caltrack_predict,
    select_best_candidate,
)
from eemeter.exceptions import (
    MissingModelParameterError,
    UnrecognizedModelTypeError,
)

@pytest.fixture
def candidate_model_no_model_status():
    return CandidateModel(
        model_type='',
        formula='formula',
        status='NO MODEL',
    )


@pytest.fixture
def candidate_model_no_model_none():
    return CandidateModel(
        model_type=None,
        formula='formula',
        status='QUALIFIED',
        predict_func=caltrack_predict,
        model_params={
            'intercept': 1,
        }
    )


def test_caltrack_predict_no_model_status(candidate_model_no_model_status):
    df = pd.DataFrame({
        'value': np.arange(30.0, 90.0),
        'n_days': 1,
    })
    with pytest.raises(ValueError):
        candidate_model_no_model_status.predict(df)


def test_caltrack_predict_no_model_none(candidate_model_no_model_none):
    df = pd.DataFrame({
        'value': np.arange(30.0, 90.0),
        'n_days': 1,
    })
    with pytest.raises(ValueError):
        candidate_model_no_model_none.predict(df)

@pytest.fixture
def candidate_model_intercept_only():
    return CandidateModel(
        model_type='intercept_only',
        formula='formula',
        status='QUALIFIED',
        predict_func=caltrack_predict,
        model_params={
            'intercept': 1,
        }
    )


def test_caltrack_predict_no_index_or_n_days(candidate_model_intercept_only):
    df = pd.DataFrame([np.arange(30, 90)]).astype(float)
    with pytest.raises(ValueError):
        candidate_model_intercept_only.predict(df)


def test_caltrack_predict_n_days(candidate_model_intercept_only):
    df = pd.DataFrame({
        'value': np.arange(30.0, 90.0),
        'n_days': 1,
    })
    prediction = candidate_model_intercept_only.predict(df)
    assert prediction.sum() == 60


@pytest.fixture
def design_matrix():
    days = pd.date_range('2011-01-01', freq='D', periods=60, tz='UTC')
    meter_data = pd.DataFrame({'value': 1}, index=days)
    temperature_data = pd.Series(np.arange(30.0, 90.0), index=days) \
        .asfreq('H').ffill()
    return merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60, 65],
        cooling_balance_points=[65, 70],
    )


def test_caltrack_predict_intercept_only(
    candidate_model_intercept_only, design_matrix
):
    prediction = candidate_model_intercept_only.predict(design_matrix)
    assert prediction.sum() == 59


def test_caltrack_predict_intercept_only_disaggregated(
    candidate_model_intercept_only, design_matrix
):
    prediction = candidate_model_intercept_only.predict(
        design_matrix, disaggregated=True
    )
    assert prediction.base_load.sum() == 59
    assert prediction.heating_load.sum() == 0
    assert prediction.cooling_load.sum() == 0


@pytest.fixture
def candidate_model_missing_params():
    return CandidateModel(
        model_type='intercept_only',
        formula='formula',
        status='QUALIFIED',
        predict_func=caltrack_predict,
        model_params={}
    )


def test_caltrack_predict_missing_params(
        candidate_model_missing_params, design_matrix):
    with pytest.raises(MissingModelParameterError):
        candidate_model_missing_params.predict(design_matrix)


@pytest.fixture
def candidate_model_cdd_only():
    return CandidateModel(
        model_type='cdd_only',
        formula='formula',
        status='QUALIFIED',
        predict_func=caltrack_predict,
        model_params={
            'intercept': 1,
            'beta_cdd': 1,
            'cooling_balance_point': 65,
        }
    )


def test_caltrack_predict_cdd_only(
    candidate_model_cdd_only, design_matrix
):
    prediction = candidate_model_cdd_only.predict(design_matrix)
    assert prediction.sum() == 335


def test_caltrack_predict_cdd_only_disaggregated(
    candidate_model_cdd_only, design_matrix
):
    prediction = candidate_model_cdd_only.predict(
        design_matrix, disaggregated=True
    )
    assert prediction.base_load.sum() == 59
    assert prediction.heating_load.sum() == 0
    assert prediction.cooling_load.sum() == 276


@pytest.fixture
def candidate_model_hdd_only():
    return CandidateModel(
        model_type='hdd_only',
        formula='formula',
        status='QUALIFIED',
        predict_func=caltrack_predict,
        model_params={
            'intercept': 1,
            'beta_hdd': 1,
            'heating_balance_point': 65,
        }
    )


def test_caltrack_predict_hdd_only(
    candidate_model_hdd_only, design_matrix
):
    prediction = candidate_model_hdd_only.predict(design_matrix)
    assert prediction.sum() == 689.0


def test_caltrack_predict_hdd_only_disaggregated(
    candidate_model_hdd_only, design_matrix
):
    prediction = candidate_model_hdd_only.predict(
        design_matrix, disaggregated=True
    )
    assert prediction.base_load.sum() == 59.0
    assert prediction.heating_load.sum() == 630.0
    assert prediction.cooling_load.sum() == 0


@pytest.fixture
def candidate_model_cdd_hdd():
    return CandidateModel(
        model_type='cdd_hdd',
        formula='formula',
        status='QUALIFIED',
        predict_func=caltrack_predict,
        model_params={
            'intercept': 1,
            'beta_hdd': 1,
            'heating_balance_point': 60,
            'beta_cdd': 1,
            'cooling_balance_point': 70,
        }
    )


def test_caltrack_predict_cdd_hdd(
    candidate_model_cdd_hdd, design_matrix
):
    prediction = candidate_model_cdd_hdd.predict(design_matrix)
    assert prediction.sum() == 695.0


def test_caltrack_predict_cdd_hdd_disaggregated(
    candidate_model_cdd_hdd, design_matrix
):
    prediction = candidate_model_cdd_hdd.predict(
        design_matrix, disaggregated=True
    )
    assert prediction.base_load.sum() == 59
    assert prediction.heating_load.sum() == 465.0
    assert prediction.cooling_load.sum() == 171.0


@pytest.fixture
def candidate_model_bad_model_type():
    return CandidateModel(
        model_type='unknown',
        formula='formula',
        status='QUALIFIED',
        predict_func=caltrack_predict,
        model_params={}
    )


def test_caltrack_predict_bad_model_type(
        candidate_model_bad_model_type, design_matrix):
    with pytest.raises(UnrecognizedModelTypeError):
        candidate_model_bad_model_type.predict(design_matrix)


def test_get_too_few_non_zero_degree_day_warning_ok():
    warnings = get_too_few_non_zero_degree_day_warning(
        model_type='model_type', balance_point=65, degree_day_type='xdd',
        degree_days=pd.Series([1, 1, 1]), minimum_non_zero=2,
    )
    assert warnings == []


def test_get_too_few_non_zero_degree_day_warning_fail():
    warnings = get_too_few_non_zero_degree_day_warning(
        model_type='model_type', balance_point=65, degree_day_type='xdd',
        degree_days=pd.Series([0, 0, 3]), minimum_non_zero=2
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.model_type.too_few_non_zero_xdd'
    )
    assert warning.description == (
        'Number of non-zero daily XDD values below accepted minimum.'
        ' Candidate fit not attempted.'
    )
    assert warning.data == {
        'minimum_non_zero_xdd': 2,
        'n_non_zero_xdd': 1,
        'xdd_balance_point': 65
    }


def test_get_total_degree_day_too_low_warning_ok():
    warnings = get_total_degree_day_too_low_warning(
        model_type='model_type', balance_point=65, degree_day_type='xdd',
        degree_days=pd.Series([1, 1, 1]), minimum_total=2
    )
    assert warnings == []


def test_get_total_degree_day_too_low_warning_fail():
    warnings = get_total_degree_day_too_low_warning(
        model_type='model_type', balance_point=65, degree_day_type='xdd',
        degree_days=pd.Series([0.5, 0.5, 0.5]), minimum_total=2
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.model_type.total_xdd_too_low'
    )
    assert warning.description == (
        'Total XDD below accepted minimum. Candidate fit not attempted.'
    )
    assert warning.data == {
        'total_xdd': 1.5,
        'total_xdd_minimum': 2,
        'xdd_balance_point': 65
    }


def test_get_parameter_negative_warning_ok():
    warnings = get_parameter_negative_warning(
        'intercept_only', {'intercept': 0}, 'intercept'
    )
    assert warnings == []


def test_get_parameter_negative_warning_fail():
    warnings = get_parameter_negative_warning(
        'intercept_only', {'intercept': -1}, 'intercept'
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.intercept_only.intercept_negative'
    )
    assert warning.description == (
        'Model fit intercept parameter is negative. Candidate model rejected.'
    )
    assert warning.data == {'intercept': -1}


def test_get_parameter_p_value_too_high_warning_ok():
    warnings = get_parameter_p_value_too_high_warning(
        'intercept_only', {'intercept': 0}, 'intercept', 0.1, 0.1
    )
    assert warnings == []


def test_get_parameter_p_value_too_high_warning_fail():
    warnings = get_parameter_p_value_too_high_warning(
        'intercept_only', {'intercept': 0}, 'intercept', 0.2, 0.1
    )
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.intercept_only.intercept_p_value_too_high'
    )
    assert warning.description == (
        'Model fit intercept p-value is too high. Candidate model rejected.'
    )
    assert warning.data == {
        'intercept': 0,
        'intercept_maximum_p_value': 0.1,
        'intercept_p_value': 0.2
    }


def test_get_intercept_only_candidate_models_qualified(design_matrix):
    data = pd.DataFrame({
        'meter_value': np.arange(10),
    })
    candidate_models = get_intercept_only_candidate_models(
        data, weights_col=None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'intercept_only'
    assert model.formula == 'meter_value ~ 1'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == ['intercept']
    assert round(model.model_params['intercept'], 2) == 4.5
    assert round(model.predict(design_matrix).sum(), 2) == 265.5
    assert model.r_squared == 0
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_intercept_only_candidate_models_qualified_with_weights(
    design_matrix
):
    data = pd.DataFrame({
        'meter_value': np.arange(10),
        'weights': np.arange(10),
    })
    candidate_models = get_intercept_only_candidate_models(
        data, 'weights')
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'intercept_only'
    assert model.formula == 'meter_value ~ 1'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == ['intercept']
    assert round(model.model_params['intercept'], 2) == 6.33
    assert round(model.predict(design_matrix).sum(), 2) == 373.67
    assert model.r_squared == 0
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_intercept_only_candidate_models_error():
    data = pd.DataFrame({
        'meter_value': [],
    })
    candidate_models = get_intercept_only_candidate_models(
        data, weights_col=None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.intercept_only.model_fit'
    )
    assert warning.description == (
        'Error encountered in statsmodels.formula.api.ols method.'
        ' (Empty data?)'
    )
    assert list(sorted(warning.data.keys())) == ['traceback']
    assert warning.data['traceback'] is not None


def test_get_cdd_only_candidate_models_qualified(design_matrix):
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, 6],
        'cdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_cdd_only_candidate_models(
        data, 1, 1, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_only'
    assert model.formula == 'meter_value ~ cdd_65'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_cdd', 'cooling_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_cdd'], 2) == 1.01
    assert round(model.model_params['cooling_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 0.97
    assert round(model.predict(design_matrix).sum(), 2) == 334.8
    assert round(model.r_squared, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_qualified_with_weights(design_matrix):
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, 6],
        'cdd_65': [0, 0.1, 0, 5],
        'weights': [1, 100, 1, 1],
    })
    candidate_models = get_cdd_only_candidate_models(
        data, 1, 1, 0.1, 'weights')
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_only'
    assert model.formula == 'meter_value ~ cdd_65'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_cdd', 'cooling_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_cdd'], 2) == 1.02
    assert round(model.model_params['cooling_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 0.9
    assert round(model.predict(design_matrix).sum(), 2) == 334.4
    assert round(model.r_squared, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_not_attempted():
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, 6],
        'cdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_cdd_only_candidate_models(
        data, 10, 10, 0.1, None,
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_only'
    assert model.formula == 'meter_value ~ cdd_65'
    assert model.status == 'NOT ATTEMPTED'
    assert model.model is None
    assert model.result is None
    assert model.model_params == {}
    with pytest.raises(ValueError):
        assert model.predict(np.ones(3))
    assert model.r_squared is None
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_disqualified(design_matrix):
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, -4],
        'cdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_cdd_only_candidate_models(
        data, 1, 1, 0.0, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_only'
    assert model.formula == 'meter_value ~ cdd_65'
    assert model.status == 'DISQUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_cdd', 'cooling_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_cdd'], 2) == -1.01
    assert round(model.model_params['cooling_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 1.03
    assert round(model.predict(design_matrix).sum(), 2) == -216.8
    assert round(model.r_squared, 2) == 1.00
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_cdd_only_candidate_models_error():
    data = pd.DataFrame({
        'meter_value': [],
        'cdd_65': [],
    })
    candidate_models = get_cdd_only_candidate_models(data, 0, 0, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.cdd_only.model_fit'
    )
    assert warning.description == (
        'Error encountered in statsmodels.formula.api.ols method.'
        ' (Empty data?)'
    )
    assert list(sorted(warning.data.keys())) == ['traceback']
    assert warning.data['traceback'] is not None


def test_get_hdd_only_candidate_models_qualified(design_matrix):
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, 6],
        'hdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_hdd_only_candidate_models(
        data, 1, 1, 0.1, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'hdd_only'
    assert model.formula == 'meter_value ~ hdd_65'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_hdd', 'heating_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_hdd'], 2) == 1.01
    assert round(model.model_params['heating_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 0.97
    assert round(model.predict(design_matrix).sum(), 2) == 691.05
    assert round(model.r_squared, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_qualified_with_weights(design_matrix):
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, 6],
        'hdd_65': [0, 0.1, 0, 5],
        'weights': [1, 100, 1, 1],
    })
    candidate_models = get_hdd_only_candidate_models(
        data, 1, 1, 0.1, 'weights'
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'hdd_only'
    assert model.formula == 'meter_value ~ hdd_65'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_hdd', 'heating_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_hdd'], 2) == 1.02
    assert round(model.model_params['heating_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 0.9
    assert round(model.predict(design_matrix).sum(), 2) == 695.18
    assert round(model.r_squared, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_not_attempted():
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, 6],
        'hdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_hdd_only_candidate_models(
        data, 10, 10, 0.1, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'hdd_only'
    assert model.formula == 'meter_value ~ hdd_65'
    assert model.status == 'NOT ATTEMPTED'
    assert model.model is None
    assert model.result is None
    assert model.model_params == {}
    with pytest.raises(ValueError):
        assert model.predict(np.ones(3))
    assert model.r_squared is None
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_disqualified(design_matrix):
    data = pd.DataFrame({
        'meter_value': [1, 1, 1, -4],
        'hdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_hdd_only_candidate_models(
        data, 1, 1, 0.0, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'hdd_only'
    assert model.formula == 'meter_value ~ hdd_65'
    assert model.status == 'DISQUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_hdd', 'heating_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_hdd'], 2) == -1.01
    assert round(model.model_params['heating_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 1.03
    assert round(model.predict(design_matrix).sum(), 2) == -573.05
    assert round(model.r_squared, 2) == 1.00
    assert len(model.warnings) == 2
    assert json.dumps(model.json()) is not None


def test_get_hdd_only_candidate_models_error():
    data = pd.DataFrame({
        'meter_value': [],
        'hdd_65': [],
    })
    candidate_models = get_hdd_only_candidate_models(
        data, 0, 0, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.hdd_only.model_fit'
    )
    assert warning.description == (
        'Error encountered in statsmodels.formula.api.ols method.'
        ' (Empty data?)'
    )
    assert list(sorted(warning.data.keys())) == ['traceback']
    assert warning.data['traceback'] is not None


def test_get_cdd_hdd_candidate_models_qualified(design_matrix):
    data = pd.DataFrame({
        'meter_value': [6, 1, 1, 6],
        'cdd_65': [5, 0, 0.1, 0],
        'hdd_65': [0, 0.1, 0.1, 5],
    })
    candidate_models = get_cdd_hdd_candidate_models(
        data, 1, 1, 1, 1, 0.1, 0.1, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_hdd'
    assert model.formula == 'meter_value ~ cdd_65 + hdd_65'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_cdd', 'beta_hdd', 'cooling_balance_point',
        'heating_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_cdd'], 2) == 1.03
    assert round(model.model_params['beta_hdd'], 2) == 1.03
    assert round(model.model_params['cooling_balance_point'], 2) == 65
    assert round(model.model_params['heating_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 0.85
    assert round(model.predict(design_matrix).sum(), 2) == 983.77
    assert round(model.r_squared, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_qualified_with_weights(design_matrix):
    data = pd.DataFrame({
        'meter_value': [6, 1, 1, 6],
        'cdd_65': [5, 0, 0.1, 0],
        'hdd_65': [0, 0.1, 0.1, 5],
        'weights': [1, 1, 100, 1],
    })
    candidate_models = get_cdd_hdd_candidate_models(
        data, 1, 1, 1, 1, 0.1, 0.1, 'weights'
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_hdd'
    assert model.formula == 'meter_value ~ cdd_65 + hdd_65'
    assert model.status == 'QUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_cdd', 'beta_hdd', 'cooling_balance_point',
        'heating_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_cdd'], 2) == 1.04
    assert round(model.model_params['beta_hdd'], 2) == 1.04
    assert round(model.model_params['cooling_balance_point'], 2) == 65
    assert round(model.model_params['heating_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 0.79
    assert round(model.predict(design_matrix).sum(), 2) == 990.2
    assert round(model.r_squared, 2) == 1.00
    assert model.warnings == []
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_not_attempted():
    data = pd.DataFrame({
        'meter_value': [6, 1, 1, 6],
        'cdd_65': [5, 0, 0.1, 0],
        'hdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_cdd_hdd_candidate_models(
        data, 10, 10, 10, 10, 0.1, 0.1, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_hdd'
    assert model.formula == 'meter_value ~ cdd_65 + hdd_65'
    assert model.status == 'NOT ATTEMPTED'
    assert model.model is None
    assert model.result is None
    assert model.model_params == {}
    with pytest.raises(ValueError):
        assert model.predict(np.ones(3))
    assert model.r_squared is None
    assert len(model.warnings) == 4
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_disqualified(design_matrix):
    data = pd.DataFrame({
        'meter_value': [-4, 1, 1, -4],
        'cdd_65': [5, 0, 0.1, 0],
        'hdd_65': [0, 0.1, 0, 5],
    })
    candidate_models = get_cdd_hdd_candidate_models(
        data, 1, 1, 1, 1, 0.0, 0.0, None
    )
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert model.model_type == 'cdd_hdd'
    assert model.formula == 'meter_value ~ cdd_65 + hdd_65'
    assert model.status == 'DISQUALIFIED'
    assert model.model is not None
    assert model.result is not None
    assert list(sorted(model.model_params.keys())) == [
        'beta_cdd', 'beta_hdd', 'cooling_balance_point',
        'heating_balance_point', 'intercept',
    ]
    assert round(model.model_params['beta_cdd'], 2) == -1.02
    assert round(model.model_params['beta_hdd'], 2) == -1.02
    assert round(model.model_params['cooling_balance_point'], 2) == 65
    assert round(model.model_params['heating_balance_point'], 2) == 65
    assert round(model.model_params['intercept'], 2) == 1.1
    assert round(model.predict(design_matrix).sum(), 2) == -859.47
    assert round(model.r_squared, 2) == 1.00
    assert len(model.warnings) == 4
    assert json.dumps(model.json()) is not None


def test_get_cdd_hdd_candidate_models_error():
    data = pd.DataFrame({
        'meter_value': [],
        'hdd_65': [],
        'cdd_65': [],
    })
    candidate_models = get_cdd_hdd_candidate_models(
        data, 0, 0, 0, 0, 0.1, 0.1, None)
    assert len(candidate_models) == 1
    model = candidate_models[0]
    assert len(model.warnings) == 1
    warning = model.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.cdd_hdd.model_fit'
    )
    assert warning.description == (
        'Error encountered in statsmodels.formula.api.ols method.'
        ' (Empty data?)'
    )
    assert list(sorted(warning.data.keys())) == ['traceback']
    assert warning.data['traceback'] is not None


@pytest.fixture
def candidate_model_qualified_high_r2():
    return CandidateModel(
        model_type='model_type',
        formula='formula1',
        status='QUALIFIED',
        r_squared=1,
    )

@pytest.fixture
def candidate_model_qualified_low_r2():
    return CandidateModel(
        model_type='model_type',
        formula='formula2',
        status='QUALIFIED',
        r_squared=0,
    )


@pytest.fixture
def candidate_model_disqualified():
    return CandidateModel(
        model_type='model_type',
        formula='formula3',
        status='DISQUALIFIED',
        r_squared=0.5,
    )


def test_select_best_candidate_ok(
    candidate_model_qualified_high_r2, candidate_model_qualified_low_r2,
    candidate_model_disqualified,
):
    candidates = [
        candidate_model_qualified_high_r2,
        candidate_model_qualified_low_r2,
        candidate_model_disqualified,
    ]

    best_candidate, warnings = select_best_candidate(candidates)
    assert warnings == []
    assert best_candidate.status == 'QUALIFIED'
    assert best_candidate.formula == 'formula1'
    assert best_candidate.r_squared == 1


def test_select_best_candidate_none(
    candidate_model_disqualified,
):
    candidates = [candidate_model_disqualified]

    best_candidate, warnings = select_best_candidate(candidates)
    assert best_candidate is None
    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.select_best_candidate.no_candidates'
    )
    assert warning.description == (
        'No qualified model candidates available.'
    )
    assert warning.data == {'status_count:DISQUALIFIED': 1}


def test_caltrack_method_empty():
    data = pd.DataFrame({
        'meter_value': [],
        'hdd_65': [],
        'cdd_65': [],
    })
    model_fit = caltrack_method(data)
    assert model_fit.method_name == 'caltrack_method'
    assert model_fit.status == 'NO DATA'
    assert len(model_fit.warnings) == 1
    warning = model_fit.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_method.no_data'
    )
    assert warning.description == (
        'No data available. Cannot fit model.'
    )
    assert warning.data == {}


@pytest.fixture
def cdd_hdd_h60_c65(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    blackout_start_date = il_electricity_cdd_hdd_daily['blackout_start_date']
    data = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[60],
        cooling_balance_points=[65],
    )
    baseline_data, warnings = get_baseline_data(
        data, end=blackout_start_date)
    return baseline_data


def test_caltrack_method_cdd_hdd(cdd_hdd_h60_c65, design_matrix):
    model_fit = caltrack_method(cdd_hdd_h60_c65)
    assert len(model_fit.candidates) == 4
    assert model_fit.candidates[0].model_type == 'intercept_only'
    assert model_fit.candidates[1].model_type == 'hdd_only'
    assert model_fit.candidates[2].model_type == 'cdd_only'
    assert model_fit.candidates[3].model_type == 'cdd_hdd'
    assert model_fit.model.status == 'QUALIFIED'
    assert model_fit.model.model_type == 'cdd_hdd'
    prediction = model_fit.model.predict(design_matrix)
    assert round(prediction.sum()) == 1629.0


def test_caltrack_method_cdd_hdd_use_billing_presets(
    cdd_hdd_h60_c65, design_matrix
):
    model_fit = caltrack_method(cdd_hdd_h60_c65, use_billing_presets=True)
    assert len(model_fit.candidates) == 4
    assert model_fit.candidates[0].model_type == 'intercept_only'
    assert model_fit.candidates[1].model_type == 'hdd_only'
    assert model_fit.candidates[2].model_type == 'cdd_only'
    assert model_fit.candidates[3].model_type == 'cdd_hdd'
    assert model_fit.model.status == 'QUALIFIED'
    assert model_fit.model.model_type == 'cdd_hdd'
    prediction = model_fit.model.predict(design_matrix)
    assert round(prediction.sum()) == 1629.0


def test_caltrack_method_no_model():
    data = pd.DataFrame({
        'meter_value': [4, 1, 1, 4],
        'cdd_65': [5, 0, 0.1, 0],
        'hdd_65': [0, 0.1, 0, 5],
    })
    model_fit = caltrack_method(
        data, fit_hdd_only=False, fit_cdd_hdd=False, fit_cdd_only=False,
        fit_intercept_only=False
    )
    assert model_fit.method_name == 'caltrack_method'
    assert model_fit.status == 'NO MODEL'
    assert len(model_fit.warnings) == 1
    warning = model_fit.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_daily.select_best_candidate.no_candidates'
    )
    assert warning.description == (
        'No qualified model candidates available.'
    )
    assert warning.data == {}


def test_caltrack_sufficiency_criteria_no_data():
    data_quality = pd.DataFrame({
        'meter_value': [],
        'temperature_not_null': [],
        'temperature_null': [],
    })
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, None, None)
    assert data_sufficiency.status == 'NO DATA'
    assert data_sufficiency.criteria_name == (
        'caltrack_sufficiency_criteria'
    )
    assert len(data_sufficiency.warnings) == 1
    warning = data_sufficiency.warnings[0]
    assert warning.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.no_data'
    )
    assert warning.description == 'No data available.'
    assert warning.data == {}


def test_caltrack_sufficiency_criteria_pass():
    data_quality = pd.DataFrame({
        'meter_value': [1, 1],
        'temperature_not_null': [1, 1],
        'temperature_null': [0, 0],
        'start': pd.date_range(
            start='2016-01-02', periods=2, freq='D').tz_localize('UTC'),
    }).set_index('start')
    requested_start = pd.Timestamp('2016-01-02') \
        .tz_localize('UTC').to_pydatetime()
    requested_end = pd.Timestamp('2016-01-03').tz_localize('UTC')
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, requested_start, requested_end, min_days=1,
        min_fraction_daily_coverage=0.9,
        min_fraction_daily_temperature_hourly_coverage=0.9,
    )
    assert data_sufficiency.status == 'PASS'
    assert data_sufficiency.criteria_name == (
        'caltrack_sufficiency_criteria'
    )
    assert data_sufficiency.warnings == []
    assert data_sufficiency.settings == {
        'min_days': 1,
        'min_fraction_daily_coverage': 0.9,
        'min_fraction_daily_temperature_hourly_coverage': 0.9
    }


def test_caltrack_sufficiency_criteria_fail_no_gap():
    data_quality = pd.DataFrame({
        'meter_value': [np.nan, 1],
        'temperature_not_null': [1, 5],
        'temperature_null': [0, 5],
        'start': pd.date_range(
            start='2016-01-02', periods=2, freq='D').tz_localize('UTC'),
    }).set_index('start')
    requested_start = pd.Timestamp('2016-01-02').tz_localize('UTC')
    requested_end = pd.Timestamp('2016-01-04').tz_localize('UTC')
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, requested_start, requested_end, min_days=3,
        min_fraction_daily_coverage=0.9,
        min_fraction_daily_temperature_hourly_coverage=0.9,
    )
    assert data_sufficiency.status == 'FAIL'
    assert data_sufficiency.criteria_name == (
        'caltrack_sufficiency_criteria'
    )
    assert len(data_sufficiency.warnings) == 4

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.too_few_total_days'
    )
    assert warning0.description == (
        'Smaller total data span than the allowable minimum.'
    )
    assert warning0.data == {'min_days': 3, 'n_days_total': 2}

    warning1 = data_sufficiency.warnings[1]
    assert warning1.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'too_many_days_with_missing_data'
    )
    assert warning1.description == (
        'Too many days in data have missing meter data or temperature data.'
    )
    assert warning1.data == {'n_days_total': 2, 'n_valid_days': 0}

    warning2 = data_sufficiency.warnings[2]
    assert warning2.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'too_many_days_with_missing_meter_data'
    )
    assert warning2.description == (
        'Too many days in data have missing meter data.'
    )
    # zero because nan value and last point dropped
    assert warning2.data == {'n_days_total': 2, 'n_valid_meter_data_days': 0}

    warning3 = data_sufficiency.warnings[3]
    assert warning3.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'too_many_days_with_missing_temperature_data'
    )
    assert warning3.description == (
        'Too many days in data have missing temperature data.'
    )
    assert warning3.data == {
        'n_days_total': 2,
        'n_valid_temperature_data_days': 1
    }


def test_caltrack_sufficiency_criteria_pass_no_requested_start_end():
    data_quality = pd.DataFrame({
        'meter_value': [1, 1],
        'temperature_not_null': [1, 1],
        'temperature_null': [0, 0],
        'start': pd.date_range(
            start='2016-01-02', periods=2, freq='D').tz_localize('UTC'),
    }).set_index('start')
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, None, None, min_days=1,
        min_fraction_daily_coverage=0.9,
        min_fraction_daily_temperature_hourly_coverage=0.9,
    )
    assert data_sufficiency.status == 'PASS'
    assert data_sufficiency.criteria_name == (
        'caltrack_sufficiency_criteria'
    )
    assert len(data_sufficiency.warnings) == 0


def test_caltrack_sufficiency_criteria_fail_with_requested_start_end():
    data_quality = pd.DataFrame({
        'meter_value': [1, 1],
        'temperature_not_null': [1, 1],
        'temperature_null': [0, 0],
        'start': pd.date_range(
            start='2016-01-02', periods=2, freq='D').tz_localize('UTC'),
    }).set_index('start')
    requested_start = pd.Timestamp('2016-01-01').tz_localize('UTC')
    requested_end = pd.Timestamp('2016-01-04').tz_localize('UTC')
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, requested_start, requested_end, min_days=2,
        min_fraction_daily_coverage=0.9,
        min_fraction_daily_temperature_hourly_coverage=0.9,
    )
    assert data_sufficiency.status == 'FAIL'
    assert data_sufficiency.criteria_name == (
        'caltrack_sufficiency_criteria'
    )
    assert len(data_sufficiency.warnings) == 3

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'too_many_days_with_missing_data'
    )
    assert warning0.description == (
        'Too many days in data have missing meter data or temperature data.'
    )
    assert warning0.data == {'n_days_total': 3, 'n_valid_days': 1}

    warning1 = data_sufficiency.warnings[1]
    assert warning1.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'too_many_days_with_missing_meter_data'
    )
    assert warning1.description == (
        'Too many days in data have missing meter data.'
    )
    assert warning1.data == {'n_days_total': 3, 'n_valid_meter_data_days': 1}

    warning2 = data_sufficiency.warnings[2]
    assert warning2.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'too_many_days_with_missing_temperature_data'
    )
    assert warning2.description == (
        'Too many days in data have missing temperature data.'
    )
    assert warning2.data == {
        'n_days_total': 3,
        'n_valid_temperature_data_days': 1
    }


def test_caltrack_sufficiency_criteria_too_much_data():
    data_quality = pd.DataFrame({
        'meter_value': [1, 1, 1],
        'temperature_not_null': [1, 1, 1],
        'temperature_null': [0, 0, 0],
        'start': pd.date_range(
            start='2016-01-02', periods=3, freq='D').tz_localize('UTC'),
    }).set_index('start')
    requested_start = pd.Timestamp('2016-01-03').tz_localize('UTC')
    requested_end = pd.Timestamp('2016-01-03').tz_localize('UTC')
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, requested_start, requested_end, min_days=2,
        min_fraction_daily_coverage=0.9,
        min_fraction_daily_temperature_hourly_coverage=0.9,
    )
    assert data_sufficiency.status == 'PASS'
    assert len(data_sufficiency.warnings) == 2

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'extra_data_after_requested_end_date'
    )
    assert warning0.description == (
        'Extra data found after requested end date.'
    )
    assert warning0.data == {
        'data_end': '2016-01-04T00:00:00+00:00',
        'requested_end': '2016-01-03T00:00:00+00:00'
    }

    warning1 = data_sufficiency.warnings[1]
    assert warning1.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.'
        'extra_data_before_requested_start_date'
    )
    assert warning1.description == (
        'Extra data found before requested start date.'
    )
    assert warning1.data == {
        'data_start': '2016-01-02T00:00:00+00:00',
        'requested_start': '2016-01-03T00:00:00+00:00'
    }


def test_caltrack_sufficiency_criteria_negative_values():
    data_quality = pd.DataFrame({
        'meter_value': [-1, 1, 1],
        'temperature_not_null': [1, 1, 1],
        'temperature_null': [0, 0, 1],
        'start': pd.date_range(
            start='2016-01-02', periods=3, freq='D').tz_localize('UTC'),
    }).set_index('start')
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, None, None, min_days=1,
        min_fraction_daily_coverage=0.9,
        min_fraction_daily_temperature_hourly_coverage=0.9,
    )
    assert data_sufficiency.status == 'FAIL'
    assert len(data_sufficiency.warnings) == 1

    warning0 = data_sufficiency.warnings[0]
    assert warning0.qualified_name == (
        'eemeter.caltrack_sufficiency_criteria.negative_meter_values'
    )
    assert warning0.description == (
        'Found negative meter data values, which may indicate presence of'
        ' solar net metering.'
    )
    assert warning0.data == {'n_negative_meter_values': 1}


def test_caltrack_sufficiency_criteria_handle_single_input():
    # just make sure this case is handled.
    data_quality = pd.DataFrame({
        'meter_value': [1],
        'temperature_not_null': [1],
        'temperature_null': [0],
        'start': pd.date_range(
            start='2016-01-02', periods=1, freq='D').tz_localize('UTC'),
    }).set_index('start')
    data_sufficiency = caltrack_sufficiency_criteria(
        data_quality, None, None, min_days=0,
        min_fraction_daily_coverage=0.9,
        min_fraction_daily_temperature_hourly_coverage=0.9,
    )

    assert data_sufficiency.status == 'FAIL'
    assert len(data_sufficiency.warnings) == 3


@pytest.fixture
def baseline_model(cdd_hdd_h60_c65):
    model_fit = caltrack_method(cdd_hdd_h60_c65)
    return model_fit.model


@pytest.fixture
def reporting_model(cdd_hdd_h60_c65):
    model_fit = caltrack_method(cdd_hdd_h60_c65)
    return model_fit.model


@pytest.fixture
def reporting_meter_data():
    index = pd.date_range('2011-01-01', freq='D', periods=60, tz='UTC')
    return pd.DataFrame({'value': 1}, index=index)


@pytest.fixture
def reporting_temperature_data():
    index = pd.date_range('2011-01-01', freq='D', periods=60, tz='UTC')
    return pd.Series(np.arange(30.0, 90.0), index=index).asfreq('H').ffill()


def test_caltrack_metered_savings_cdd_hdd(
    baseline_model, reporting_meter_data, reporting_temperature_data
):

    results = caltrack_metered_savings(
        baseline_model, reporting_meter_data, reporting_temperature_data,
        degree_day_method='daily'
    )
    assert list(results.columns) == [
        'reporting_observed', 'counterfactual_usage', 'metered_savings'
    ]
    assert round(results.metered_savings.sum(), 2) == 1569.57


def test_caltrack_metered_savings_cdd_hdd_hourly_degree_days(
    baseline_model, reporting_meter_data, reporting_temperature_data
):

    results = caltrack_metered_savings(
        baseline_model, reporting_meter_data, reporting_temperature_data,
        degree_day_method='hourly'
    )
    assert list(results.columns) == [
        'reporting_observed', 'counterfactual_usage', 'metered_savings'
    ]
    assert round(results.metered_savings.sum(), 2) == 1569.57


def test_caltrack_metered_savings_cdd_hdd_no_params(
    baseline_model, reporting_meter_data, reporting_temperature_data
):
    baseline_model.model_params = None
    with pytest.raises(MissingModelParameterError):
        caltrack_metered_savings(
            baseline_model, reporting_meter_data, reporting_temperature_data,
            degree_day_method='daily'
        )


def test_caltrack_metered_savings_cdd_hdd_with_disaggregated(
    baseline_model, reporting_meter_data, reporting_temperature_data
):

    results = caltrack_metered_savings(
        baseline_model, reporting_meter_data, reporting_temperature_data,
        degree_day_method='daily', with_disaggregated=True
    )
    assert list(results.columns) == [
        'reporting_observed',
        'counterfactual_usage',
        'metered_savings',
        'counterfactual_base_load',
        'counterfactual_cooling_load',
        'counterfactual_heating_load'
    ]


def test_caltrack_modeled_savings_cdd_hdd(
    baseline_model, reporting_model, reporting_meter_data,
    reporting_temperature_data
):
    # using reporting data for convenience, but intention is to use normal data
    results = caltrack_modeled_savings(
        baseline_model, reporting_model, reporting_meter_data.index,
        reporting_temperature_data, degree_day_method='daily'
    )
    assert list(results.columns) == [
        'modeled_baseline_usage', 'modeled_reporting_usage', 'modeled_savings'
    ]
    assert round(results.modeled_savings.sum(), 2) == 0.0


def test_caltrack_modeled_savings_cdd_hdd_hourly_degree_days(
    baseline_model, reporting_model, reporting_meter_data,
    reporting_temperature_data
):
    # using reporting data for convenience, but intention is to use normal data
    results = caltrack_modeled_savings(
        baseline_model, reporting_model, reporting_meter_data.index,
        reporting_temperature_data, degree_day_method='hourly'
    )
    assert list(results.columns) == [
        'modeled_baseline_usage', 'modeled_reporting_usage', 'modeled_savings'
    ]
    assert round(results.modeled_savings.sum(), 2) == 0.0


def test_caltrack_modeled_savings_cdd_hdd_baseline_model_no_params(
    baseline_model, reporting_model, reporting_meter_data,
    reporting_temperature_data
):
    baseline_model.model_params = None
    with pytest.raises(MissingModelParameterError):
        caltrack_modeled_savings(
            baseline_model, reporting_model, reporting_meter_data.index,
            reporting_temperature_data, degree_day_method='daily'
        )


def test_caltrack_modeled_savings_cdd_hdd_reporting_model_no_params(
    baseline_model, reporting_model, reporting_meter_data,
    reporting_temperature_data
):
    reporting_model.model_params = None
    with pytest.raises(MissingModelParameterError):
        caltrack_modeled_savings(
            baseline_model, reporting_model, reporting_meter_data.index,
            reporting_temperature_data, degree_day_method='daily'
        )


def test_caltrack_modeled_savings_cdd_hdd_with_disaggregated(
    baseline_model, reporting_model, reporting_meter_data,
    reporting_temperature_data
):

    # using reporting data for convenience, but intention is to use normal data
    results = caltrack_modeled_savings(
        baseline_model, reporting_model, reporting_meter_data.index,
        reporting_temperature_data, degree_day_method='daily',
        with_disaggregated=True
    )
    assert list(results.columns) == [
        'modeled_baseline_usage',
        'modeled_reporting_usage',
        'modeled_savings',
        'modeled_baseline_base_load',
        'modeled_baseline_cooling_load',
        'modeled_baseline_heating_load',
        'modeled_reporting_base_load',
        'modeled_reporting_cooling_load',
        'modeled_reporting_heating_load',
        'modeled_base_load_savings',
        'modeled_heating_load_savings',
        'modeled_cooling_load_savings'
    ]
