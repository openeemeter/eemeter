import json
import pytest
import pandas as pd
import numpy as np

from eemeter import (
    ModelMetrics,
)
from eemeter.metrics import (
    _compute_r_squared,
    _compute_r_squared_adj,
    _compute_cvrmse,
    _compute_cvrmse_adj,
    _compute_mape,
    _compute_nmae,
    _compute_nmbe,
    _compute_autocorr_resid,
)


@pytest.fixture
def sample_data():
    # Could have included DatetimeIndex, but made it more general
    series_one = pd.Series([1, 3, 4, 1, 6])
    series_two =  pd.Series([2, 3, 3, 2, 4])
    series_one.name = "NameOne" 
    series_two.name = "NameTwo"
    return series_one, series_two

    
def test_ModelMetrics(sample_data):
    series_one, series_two = sample_data
    model_metrics = ModelMetrics(series_one,series_two,num_parameters=2)
    assert model_metrics.observed_length == 5
    assert model_metrics.predicted_length == 5
    assert model_metrics.merged_length == 5
    assert model_metrics.observed_mean == 3.0
    assert model_metrics.predicted_mean == 2.8
    assert round(model_metrics.observed_skew, 3) == 0.524 
    assert round(model_metrics.predicted_skew, 3) == 0.512
    assert round(model_metrics.observed_kurtosis, 3) == -0.963
    assert round(model_metrics.predicted_kurtosis, 3) == -0.612
    assert round(model_metrics.observed_cvstd, 3) == 0.707
    assert round(model_metrics.predicted_cvstd, 3) == 0.299
    assert round(model_metrics.r_squared, 3) == 0.972
    assert round(model_metrics.r_squared_adj, 3) == 0.944
    assert round(model_metrics.cvrmse, 3) == 0.394
    assert round(model_metrics.cvrmse_adj, 3) == 0.509
    assert round(model_metrics.mape, 3) == 0.517
    assert round(model_metrics.mape_no_zeros, 3) == 0.517
    assert model_metrics.num_meter_zeros == 0
    assert round(model_metrics.nmae, 3) == 0.333
    assert round(model_metrics.nmbe, 3) == -0.067
    assert round(model_metrics.autocorr_resid, 3) == -0.674
    assert repr(model_metrics) is not None
    assert json.dumps(model_metrics.json()) is not None


@pytest.fixture
def sample_data_zeros():
    series_one = pd.Series([1, 0, 0, 1, 6])
    series_two =  pd.Series([2, 3, 3, 2, 4])
    return series_one, series_two


def test_ModelMetrics_zeros(sample_data_zeros):
    series_one, series_two = sample_data_zeros
    model_metrics = ModelMetrics(series_one, series_two, num_parameters = 2)
    assert np.isinf(model_metrics.mape)
    assert model_metrics.num_meter_zeros == 2

    
def test_ModelMetrics_num_parameter_error(sample_data):
    series_one, series_two = sample_data
    with pytest.raises(ValueError):
        model_metrics = ModelMetrics(series_one, series_two, num_parameters = -1)

    
def test_ModelMetrics_autocorr_lags_error(sample_data):
    series_one, series_two = sample_data
    with pytest.raises(ValueError):
        model_metrics = ModelMetrics(series_one, series_two, autocorr_lags = 0)

        
@pytest.fixture
def sample_data_diff_length():
    series_one = pd.Series([1, 0, 0, 1, 6, 4, 5])
    series_two =  pd.Series([2, 3, 3, 2, 4])
    return series_one, series_two
        

def test_ModelMetrics_diff_length_error(sample_data_diff_length):
    series_one, series_two = sample_data_diff_length
    with pytest.raises(ValueError):
        model_metrics = ModelMetrics(series_one, series_two)
    
    
def test_ModelMetrics_inputs_unchanged(sample_data):
    series_one, series_two = sample_data
    model_metrics = ModelMetrics(series_one, series_two)
    assert sample_data[0].name == "NameOne"
    assert sample_data[1].name == "NameTwo"
    
    
@pytest.fixture
def sample_data_merged(sample_data):
    series_one, series_two = sample_data
    observed = series_one.to_frame().dropna()
    predicted = series_two.to_frame().dropna()
    observed.columns = ['observed']
    predicted.columns = ['predicted']
    combined = observed.merge(predicted, left_index=True, 
        right_index=True)
    combined['residuals'] = (combined.predicted - combined.observed)    
    return combined
 
    
def test_compute_r_squared(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_r_squared(combined), 3) == 0.972
    
    
def test_compute_r_squared_adj(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_r_squared_adj(_compute_r_squared(combined), 5, 2), 3) == 0.944
    
    
def test_compute_cvrmse(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_cvrmse(combined), 3) == 0.394
    
    
def test_compute_cvrmse_adj(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_cvrmse_adj(combined, 5, 2), 3) == 0.509
    
    
def test_compute_mape(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_mape(combined), 3) == 0.517
        
    
def test_compute_nmae(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_nmae(combined), 3) == 0.333
    
    
def test_compute_nmbe(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_nmbe(combined), 3) == -0.067
    
    
def test_compute_autocorr_resid(sample_data_merged):
    combined = sample_data_merged
    assert round(_compute_autocorr_resid(combined, 1), 3) == -0.674