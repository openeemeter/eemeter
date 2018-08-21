import pytest
import pandas as pd

from eemeter import (
    ModelMetrics,
)
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

@pytest.fixture
def cdd_hdd_h53_c68(il_electricity_cdd_hdd_daily):
    meter_data = il_electricity_cdd_hdd_daily['meter_data']
    temperature_data = il_electricity_cdd_hdd_daily['temperature_data']
    blackout_start_date = il_electricity_cdd_hdd_daily['blackout_start_date']
    baseline_meter_data_daily, baseline_warnings_daily = get_baseline_data(
    meter_data, end=blackout_start_date, max_days=365)
    data = merge_temperature_data(
        meter_data, temperature_data,
        heating_balance_points=[53],
        cooling_balance_points=[68],
    )
    baseline_model_fit = caltrack_method(
        data,
    )
    predicted_data = caltrack_predict(
        'cdd_hdd', baseline_model_fit.model.model_params, \
        temperature_data, data.index,'daily'
    )
    return meter_data.meter_data, predicted_data.predicted_usage
    
def test_ModelMetrics(cdd_hdd_h53_c68):
    observed, predicted = cdd_hdd_h53_c68
    metrics_result = ModelMetrics(observed, predicted, 2, 1)
    assert round(metrics_result.cvrmse, 3) == .262
    
    
