from eemeter.models import TemperatureSensitivityModel

from numpy.testing import assert_almost_equal
import numpy as np

import pytest

def test_TemperatureSensitivityModel_with_heating_and_cooling():
    initial_params = {
        "base_consumption": -1,
        "heating_slope": 0,
        "cooling_slope": 0,
        "heating_reference_temperature": 55,
        "cooling_reference_temperature": 57,
    }
    param_bounds = {
        "base_consumption": [0,100],
        "heating_slope": [0,100],
        "cooling_slope": [0,100],
        "heating_reference_temperature": [50,60],
        "cooling_reference_temperature": [52,72],
    }
    model = TemperatureSensitivityModel(heating=True,cooling=True)
    model.initial_params = None
    model.param_bounds = None
    model = TemperatureSensitivityModel(True,True,initial_params,param_bounds)

    params = [1,1,60,1,65]
    observed_temps = np.array([[i] for i in range(50,70)])
    usages = model.compute_usage_estimates(params,observed_temps)
    assert_almost_equal(usages[8:18],[3,2,1,1,1,1,1,1,2,3])
    opt_params = model.parameter_optimization(usages, observed_temps)
    assert_almost_equal(params,opt_params,decimal=3)

def test_TemperatureSensitivityModel_with_heating():
    initial_params = {
        "base_consumption": 0,
        "heating_slope": 0,
        "heating_reference_temperature": 55,
    }
    param_bounds = {
        "base_consumption": [0,100],
        "heating_slope": [0,100],
        "heating_reference_temperature": [50,60],
    }
    model = TemperatureSensitivityModel(heating=True,cooling=False,initial_params=initial_params,param_bounds=param_bounds)
    params = [1,1,60]
    observed_temps = np.array([[i] for i in range(50,70)])
    usages = model.compute_usage_estimates(params,observed_temps)
    assert_almost_equal(usages[8:13],[3,2,1,1,1])
    opt_params = model.parameter_optimization(usages, observed_temps)
    assert_almost_equal(params,opt_params,decimal=3)

def test_TemperatureSensitivityModel_with_cooling():
    initial_params = {
        "base_consumption": 0,
        "cooling_slope": 0,
        "cooling_reference_temperature": 57,
    }
    param_bounds = {
        "base_consumption": [0,100],
        "cooling_slope": [0,100],
        "cooling_reference_temperature": [52,72],
    }
    model = TemperatureSensitivityModel(heating=False,cooling=True,initial_params=initial_params,param_bounds=param_bounds)
    params = [1,1,60]
    observed_temps = np.array([[i] for i in range(50,70)])
    usages = model.compute_usage_estimates(params,observed_temps)
    assert_almost_equal(usages[8:13],[1,1,1,2,3])
    opt_params = model.parameter_optimization(usages, observed_temps)
    assert_almost_equal(params,opt_params,decimal=3)
