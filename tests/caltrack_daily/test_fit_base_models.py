import numpy as np
import pandas as pd
import pytest
from eemeter.caltrack.daily.utilities.config import FullModelSelection
from eemeter.caltrack.daily.utilities.config import DailySettings as Settings
from eemeter.caltrack.daily.utilities.utils import ModelCoefficients
from eemeter.caltrack.daily.fit_base_models import (
    fit_initial_models_from_full_model, fit_model, fit_final_model, _get_opt_options
)

from eemeter.caltrack.daily.optimize_results import OptimizedResult

@pytest.fixture
def meter_data():
    df_meter = pd.DataFrame({
        "temperature": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0],
        "observed": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0],
        "datetime": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06", "2022-01-07", "2022-01-08", "2022-01-09", "2022-01-10", "2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14", "2022-01-15", "2022-01-16", "2022-01-17", "2022-01-18", "2022-01-19"],
    })
    df_meter["datetime"] = pd.to_datetime(df_meter["datetime"])
    df_meter.set_index("datetime", inplace=True)
    return df_meter

@pytest.fixture
def get_settings():
    return Settings(developer_mode = True, alpha_final_type = None, final_bounds_scalar = None)

def test_fit_initial_models_from_full_model(meter_data, get_settings):
    # Test case 1: Test the function with a sample dataset
    model_res = fit_initial_models_from_full_model(meter_data, get_settings)
    assert isinstance(model_res, OptimizedResult)
    
    # Test case 3: Test the function with a dataset that has missing values
    T = np.array([10, 20, 30, 40, 50])
    obs = np.array([1, 2, np.nan, 4, 5])
    model_res = fit_initial_models_from_full_model(meter_data, get_settings)
    assert isinstance(model_res, OptimizedResult)
    
    # Test case 4: Test the function with a dataset that has negative values
    T = np.array([10, 20, 30, 40, 50])
    obs = np.array([-1, 2, 3, 4, 5])
    model_res = fit_initial_models_from_full_model(meter_data, get_settings)
    assert isinstance(model_res, OptimizedResult)

def test_fit_model(meter_data,get_settings):
    # Test case 1: Test for model_key = "hdd_tidd_cdd_smooth"
    T = np.array(meter_data["temperature"])
    obs = np.array(meter_data["observed"])

    fit_input = [T, obs, get_settings, _get_opt_options(get_settings)]
    res = fit_model("hdd_tidd_cdd_smooth", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # Test case 2: Test for model_key = "hdd_tidd_cdd"
    res = fit_model("hdd_tidd_cdd", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # Test case 3: Test for model_key = "c_hdd_tidd_smooth"
    res = fit_model("c_hdd_tidd_smooth", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # Test case 4: Test for model_key = "c_hdd_tidd"
    res = fit_model("c_hdd_tidd", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # Test case 5: Test for model_key = "tidd"
    res = fit_model("tidd", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

def test_fit_final_model(get_settings):
    # Test case 1: Test if the function returns an instance of OptimizedResult
    df_meter = {"temperature": np.array([10, 20, 30]), "observed": np.array([1, 2, 3])}
    HoF = OptimizedResult(
        model_key="tidd",
        named_coeffs={"alpha": 1.0, "beta": 2.0},
        x=np.array([1.0, 2.0]),
        loss_alpha=0.0,
        time_elapsed=0.0,
    )
    settings = {"algorithm_choice": "differential_evolution", "final_bounds_scalar": 0.1}
    res = fit_final_model(df_meter, HoF, settings)
    assert isinstance(res, OptimizedResult)

    # Test case 2: Test if the function raises a TypeError when the input arguments are of the wrong type
    with pytest.raises(TypeError):
        fit_final_model("not a dataframe", "not an OptimizedResult", "not a dictionary")

    # Test case 3: Test if the function returns an OptimizedResult with the expected loss_alpha value
    df_meter = {"temperature": np.array([10, 20, 30]), "observed": np.array([1, 2, 3])}
    HoF = OptimizedResult(
        model_key="tidd",
        named_coeffs={"alpha": 1.0, "beta": 2.0},
        x=np.array([1.0, 2.0]),
        loss_alpha=0.0,
        time_elapsed=0.0,
    )
    settings = {"algorithm_choice": "differential_evolution", "final_bounds_scalar": 0.1}
    res = fit_final_model(df_meter, HoF, settings)
    assert res.loss_alpha != 0.0