import numpy as np

from eemeter.caltrack.daily.objective_function import (
    get_idx, no_weights_obj_fcn
)


def test_get_idx():
    # Test case 1: Test with empty lists
    A = []
    B = []
    assert get_idx(A, B) == []

    # Test case 2: Test with one empty list
    A = [1, 2, 3]
    B = []
    assert get_idx(A, B) == []

    # Test case 3: Test with one non-empty list
    A = [1, 2, 3]
    B = [1, 2, 3, 4, 5]
    assert get_idx(A, B) == [0, 1, 2]

    # Test case 4: Test with two non-empty lists
    A = ["a", "b", "c"]
    B = ["a1", "b2", "c3", "d4", "e5"]
    assert get_idx(A, B) == [0, 1, 2]

    # Test case 5: Test with two non-empty lists with duplicates
    A = ["a", "b", "c"]
    B = ["a1", "b2", "c3", "a4", "e5"]
    assert get_idx(A, B) == [0, 1, 2, 3]

def test_no_weights_obj_fcn():
    # Test case 1: Test with X, obs and idx_bp as None
    X = None
    obs = None
    idx_bp = None
    model_fcn = lambda x: x
    aux_inputs = (model_fcn, obs, idx_bp)
    assert no_weights_obj_fcn(X, aux_inputs) == 0

    # Test case 2: Test with X, obs and idx_bp as empty arrays
    X = np.array([])
    obs = np.array([])
    idx_bp = np.array([])
    model_fcn = lambda x: x
    aux_inputs = (model_fcn, obs, idx_bp)
    assert no_weights_obj_fcn(X, aux_inputs) == 0

    # Test case 3: Test with X, obs and idx_bp as non-empty arrays
    X = np.array([1, 2, 3])
    obs = np.array([2, 4, 6])
    idx_bp = np.array([0, 2])
    model_fcn = lambda x: x * 2
    aux_inputs = (model_fcn, obs, idx_bp)
    assert no_weights_obj_fcn(X, aux_inputs) == 0

    # Test case 4: Test with X, obs and idx_bp as non-empty arrays with negative values
    X = np.array([-1, -2, -3])
    obs = np.array([-2, -4, -6])
    idx_bp = np.array([0, 2])
    model_fcn = lambda x: x * -2
    aux_inputs = (model_fcn, obs, idx_bp)
    assert no_weights_obj_fcn(X, aux_inputs) == 0