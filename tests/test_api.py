import pytest


from eemeter import (
    CandidateModel,
    DataSufficiency,
    EEMeterWarning,
    ModelFit,
)


def test_candidate_model_minimal():
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='status',
    )
    assert candidate_model.model_type == 'model_type'
    assert candidate_model.formula == 'formula'
    assert candidate_model.status == 'status'
    with pytest.raises(ValueError):
        candidate_model.predict('a') == 'a'
    with pytest.raises(ValueError):
        candidate_model.predict_index('a') == 'a'
    assert candidate_model.model_params == {}
    assert candidate_model.warnings == []
    assert str(candidate_model).startswith('CandidateModel')
    assert candidate_model.json() == {
        'formula': 'formula',
        'model_params': {},
        'model_type': 'model_type',
        'r_squared': None,
        'status': 'status',
        'warnings': []
    }


def test_candidate_model_with_predict_func():
    def predict_func(model_type, model_params, input_):
        return input_
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='status',
        predict_func=predict_func,
    )
    assert candidate_model.model_type == 'model_type'
    assert candidate_model.formula == 'formula'
    assert candidate_model.status == 'status'
    assert candidate_model.predict('a') == 'a'


def test_candidate_model_with_no_plot_func():
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='status',
    )
    with pytest.raises(ValueError):
        candidate_model.plot('a')


def test_candidate_model_json_with_warning():
    eemeter_warning = EEMeterWarning(
        qualified_name='qualified_name',
        description='description',
        data={}
    )
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='status',
        warnings=[eemeter_warning]
    )
    assert candidate_model.json() == {
        'formula': 'formula',
        'model_params': {},
        'model_type': 'model_type',
        'r_squared': None,
        'status': 'status',
        'warnings': [
            {
                'data': {},
                'description': 'description',
                'qualified_name': 'qualified_name',
            },
        ],
    }


def test_data_sufficiency_minimal():
    data_sufficiency = DataSufficiency(
        status='status',
        criteria_name='criteria_name',
    )
    assert data_sufficiency.status == 'status'
    assert data_sufficiency.criteria_name == 'criteria_name'
    assert data_sufficiency.warnings == []
    assert data_sufficiency.settings == {}
    assert str(data_sufficiency).startswith('DataSufficiency')
    assert data_sufficiency.json() == {
        'criteria_name': 'criteria_name',
        'settings': {},
        'status': 'status',
        'warnings': [],
    }


def test_data_sufficiency_json_with_warning():
    eemeter_warning = EEMeterWarning(
        qualified_name='qualified_name',
        description='description',
        data={}
    )
    data_sufficiency = DataSufficiency(
        status='status',
        criteria_name='criteria_name',
        warnings=[eemeter_warning]
    )
    assert data_sufficiency.json() == {
        'criteria_name': 'criteria_name',
        'settings': {},
        'status': 'status',
        'warnings': [
            {
                'data': {},
                'description': 'description',
                'qualified_name': 'qualified_name',
            },
        ],
    }


def test_eemeter_warning():
    eemeter_warning = EEMeterWarning(
        qualified_name='qualified_name',
        description='description',
        data={}
    )
    assert eemeter_warning.qualified_name == 'qualified_name'
    assert eemeter_warning.description == 'description'
    assert eemeter_warning.data == {}
    assert str(eemeter_warning).startswith('EEMeterWarning')
    assert eemeter_warning.json() == {
        'data': {},
        'description': 'description',
        'qualified_name': 'qualified_name',
    }


def test_model_fit_minimal():
    model_fit = ModelFit(
        status='status',
        method_name='method_name',
    )
    assert model_fit.status == 'status'
    assert model_fit.method_name == 'method_name'
    assert model_fit.model is None
    assert model_fit.r_squared is None
    assert model_fit.candidates == []
    assert model_fit.warnings == []
    assert model_fit.metadata == {}
    assert model_fit.settings == {}
    assert str(model_fit).startswith('ModelFit')
    assert model_fit.json() == {
        'metadata': {},
        'method_name': 'method_name',
        'model': None,
        'settings': {},
        'status': 'status',
        'r_squared': None,
        'warnings': [],
    }


def test_model_fit_json_with_objects():
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='status',
    )
    eemeter_warning = EEMeterWarning(
        qualified_name='qualified_name',
        description='description',
        data={}
    )
    model_fit = ModelFit(
        status='status',
        method_name='method_name',
        model=candidate_model,
        candidates=[candidate_model],
        warnings=[eemeter_warning],
    )
    assert model_fit.json(with_candidates=True) == {
        'candidates': [{
            'formula': 'formula',
            'model_params': {},
            'model_type': 'model_type',
            'r_squared': None,
            'status': 'status',
            'warnings': []
        }],
        'metadata': {},
        'method_name': 'method_name',
        'model': {
            'formula': 'formula',
            'model_params': {},
            'model_type': 'model_type',
            'r_squared': None,
            'status': 'status',
            'warnings': []
        },
        'settings': {},
        'status': 'status',
        'r_squared': None,
        'warnings': [{
            'data': {},
            'description': 'description',
            'qualified_name': 'qualified_name',
        }],
    }
