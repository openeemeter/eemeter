from eemeter.models import AverageDailyTemperatureSensitivityModel

from numpy.testing import assert_allclose
import numpy as np

import pytest

def test_average_daily_baseload_heating_cooling_consumption_model():
    initial_params = {
        "base_daily_consumption": 0,
        "heating_slope": 0,
        "cooling_slope": 0,
        "heating_balance_temperature": 55,
        "cooling_balance_temperature": 57,
    }
    param_bounds = {
        "base_daily_consumption": [0,100],
        "heating_slope": [0,100],
        "cooling_slope": [0,100],
        "heating_balance_temperature": [50,60],
        "cooling_balance_temperature": [52,72],
    }

    model = AverageDailyTemperatureSensitivityModel(cooling=True, heating=True, initial_params=initial_params, param_bounds=param_bounds)

    params = model.param_type([1,60,1,65,1])
    observed_temps = np.array([[i] for i in range(50,70)])
    usages = model.transform(observed_temps, params)
    assert_allclose(usages[8:18],[3,2,1,1,1,1,1,1,2,3], rtol=1e-2, atol=1e-2)
    opt_params = model.fit(observed_temps, usages)
    assert_allclose(params.to_list(), opt_params.to_list(), rtol=1e-2, atol=1e-2)

def test_TemperatureSensitivityModel_with_heating():
    initial_params = {
        "base_daily_consumption": 0,
        "heating_slope": 0,
        "heating_balance_temperature": 55,
    }
    param_bounds = {
        "base_daily_consumption": [0,100],
        "heating_slope": [0,100],
        "heating_balance_temperature": [50,60],
    }
    model = AverageDailyTemperatureSensitivityModel(heating=True, cooling=False, initial_params=initial_params, param_bounds=param_bounds)
    params = model.param_type([1,60,1])
    observed_temps = np.array([[i] for i in range(50,70)])
    usages = model.transform(observed_temps,params)
    assert_allclose(usages[8:13],[3,2,1,1,1], rtol=1e-2, atol=1e-2)
    opt_params = model.fit(observed_temps, usages)
    assert_allclose(params.to_list(), opt_params.to_list(), rtol=1e-2, atol=1e-2)

def test_TemperatureSensitivityModel_with_cooling():
    initial_params = {
        "base_daily_consumption": 0,
        "cooling_slope": 0,
        "cooling_balance_temperature": 57,
    }
    param_bounds = {
        "base_daily_consumption": [0,100],
        "cooling_slope": [0,100],
        "cooling_balance_temperature": [52,72],
    }
    model = AverageDailyTemperatureSensitivityModel(heating=False,cooling=True,initial_params=initial_params,param_bounds=param_bounds)
    params = model.param_type([1,60,1])
    observed_temps = np.array([[i] for i in range(50,70)])
    usages = model.transform(observed_temps, params)
    assert_allclose(usages[8:13],[1,1,1,2,3])
    opt_params = model.fit(observed_temps, usages)
    assert_allclose(params.to_list(), opt_params.to_list(), rtol=1e-2, atol=1e-2)

def test_model_weather_input_not_np_array():
    model = AverageDailyTemperatureSensitivityModel(heating=False,cooling=False)
    params = model.param_type([1])
    observed_temps = [[70],[65,60]]
    usages = model.transform(observed_temps, params)
    assert_allclose(usages, [1,1], rtol=1e-2, atol=1e-2)
