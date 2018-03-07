import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')

from eemeter import (
    CandidateModel,
    ModelFit,
    plot_candidate,
    plot_energy_signature,
    plot_model_fit,
    plot_time_series,
)


@pytest.fixture
def predict_func():
    def _predict_func(model_type, model_params, input_):
        return input_
    return _predict_func


def test_plot_time_series(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    ax_m, ax_t = plot_time_series(meter_data, temperature_data)
    m_data = ax_m.lines[0].get_xydata()
    t_data = ax_t.lines[0].get_xydata()
    assert m_data.shape == (810, 2)
    assert t_data.shape == (19417, 2)


def test_plot_energy_signature(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    ax = plot_energy_signature(meter_data, temperature_data, title='title')
    data = ax.collections[0].get_offsets()
    assert data.shape == (810, 2)
    assert ax.get_title() == 'title'


def test_plot_candidate_qualified(predict_func):
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='QUALIFIED',
        predict_func=predict_func,
    )
    ax = plot_candidate(candidate_model, title='title')
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)
    assert ax.get_title() == 'title'


def test_plot_candidate_disqualified(predict_func):
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='DISQUALIFIED',
        predict_func=predict_func,
    )
    ax = plot_candidate(candidate_model)
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)


def test_plot_candidate_with_range(predict_func):
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='QUALIFIED',
        predict_func=predict_func,
    )
    ax = plot_candidate(candidate_model, temp_range=(10, 20))
    data = ax.lines[0].get_xydata()
    assert data.shape == (10, 2)


def test_plot_candidate_best(predict_func):
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='QUALIFIED',
        predict_func=predict_func,
    )
    ax = plot_candidate(candidate_model, best=True)
    data = ax.lines[0].get_xydata()
    assert data.shape == (60, 2)


def test_plot_candidate_error():
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='ERROR',
    )
    ax = plot_candidate(candidate_model)
    assert ax is None


def test_plot_model_fit(il_electricity_cdd_hdd_daily, predict_func):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    candidate_model = CandidateModel(
        model_type='model_type',
        formula='formula',
        status='QUALIFIED',
        predict_func=predict_func,
    )
    model_fit = ModelFit(
        status='status',
        method_name='method_name',
        model=candidate_model,
        candidates=[candidate_model],
    )
    ax = plot_model_fit(
        meter_data, temperature_data, model_fit, title='title',
        with_candidates=True)
    data = ax.lines[0].get_xydata()
    assert data.shape == (91, 2)
    data = ax.lines[1].get_xydata()
    assert data.shape == (91, 2)
    data = ax.collections[0].get_offsets()
    assert data.shape == (810, 2)
    assert ax.get_title() == 'title'
