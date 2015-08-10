from eemeter.config.yaml_parser import load

from eemeter.meter import DataCollection

from numpy.testing import assert_allclose
import numpy as np

RTOL = 1e-2
ATOL = 1e-2

import pytest

def test_cvrmse():
    meter_yaml = """
        !obj:eemeter.meter.CVRMSE {
            input_mapping: { "y": {}, "y_hat": {}, "params": {}},
            output_mapping: { "cvrmse": {}}
        }
    """
    meter = load(meter_yaml)

    data_collection = DataCollection(
            y=np.array([12,13,414,12,23,12,32,np.nan]),
            y_hat=np.array([32,12,322,21,22,41,32,np.nan]),
            params=np.array([1,3,4]))
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data("cvrmse").value, 59.79,
            rtol=RTOL, atol=ATOL)

def test_rmse():
    meter_yaml = """
        !obj:eemeter.meter.RMSE {
            input_mapping: { "y": {}, "y_hat": {} },
            output_mapping: { "rmse": {}}
        }
    """
    meter = load(meter_yaml)

    data_collection = DataCollection(
            y=np.array([12,13,414,12,23,12,32,np.nan]),
            y_hat=np.array([32,12,322,21,22,41,32,np.nan]))
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data("rmse").value, 34.97,
            rtol=RTOL, atol=ATOL)

def test_r_squared():
    meter_yaml = """
        !obj:eemeter.meter.RSquared {
            input_mapping: { "y": {}, "y_hat": {}},
            output_mapping: { "r_squared": {}}
        }
    """
    meter = load(meter_yaml)

    data_collection = DataCollection(
            y=np.array([12,13,414,12,23,12,32,np.nan]),
            y_hat=np.array([32,12,322,21,22,41,32,np.nan]))
    result = meter.evaluate(data_collection)

    assert_allclose(result.get_data("r_squared").value, 0.9276,
            rtol=RTOL, atol=ATOL)
