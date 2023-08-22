import pytest
import numpy as np

from eemeter.caltrack.daily.optimize_results import (
    get_k, reduce_model, acf, OptimizedResult
)


from eemeter.caltrack.daily.utilities.utils import ModelCoefficients
from eemeter.caltrack.daily.utilities.config import DailySettings as Settings

def test_get_k():
    # Test case 1: Test when all values are within the bounds
    X = [60, 0.5, 80, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [70.0, 10.0, 74.0, 6.0]

    # Test case 2: Test when hdd_bp is greater than T_max_seg
    X = [100, 0.5, 80, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [100, 0.0, 86.0, -6.0]

    # Test case 3: Test when cdd_bp is less than T_min_seg
    X = [60, 0.5, 20, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [40.0, -20.0, 20, 0.0]

    # Test case 4: Test when both hdd_k and cdd_k are zero
    X = [100, 0.5, 20, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [20, 0.0, 20, 0.0]

@pytest.mark.parametrize(
    "hdd_bp, hdd_beta, pct_hdd_k, cdd_bp, cdd_beta, pct_cdd_k, intercept, T_min, T_max, T_min_seg, T_max_seg, model_key, expected_coef_id, expected_x",
    [
        # Test case 1
        (
            10, 20, 30, 40, 50, 60, 70, 0, 100, 20, 80,
            "hdd_tidd_cdd_smooth",
            ['hdd_bp', 'hdd_beta', 'hdd_k', 'cdd_bp', 'cdd_beta', 'cdd_k', 'intercept'],
            [10, -20, 30, 40, 50, 60, 70],
        ),
        # Test case 2
        (
            10, 20, 0, 40, 50, 60, 70, 0, 100, 20, 80,
            "hdd_tidd_cdd_smooth",
            ['hdd_bp', 'hdd_beta', 'hdd_k', 'cdd_bp', 'cdd_beta', 'cdd_k', 'intercept'],
            [40, 50, 60, 70],
        ),
        # Test case 3
        (
            10,
            0,
            30,
            40,
            50,
            60,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["c_hdd_bp", "c_hdd_beta", "intercept"],
            [10, 0, 70],
        ),
        # Test case 4
        (
            10,
            20,
            0,
            40,
            50,
            0,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["c_hdd_bp", "c_hdd_beta", "intercept"],
            [40, 50, 70],
        ),
        # Test case 5
        (
            10,
            0,
            0,
            40,
            0,
            0,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["intercept"],
            [70],
        ),
    ],
)
def test_reduce_model(
    hdd_bp,
    hdd_beta,
    pct_hdd_k,
    cdd_bp,
    cdd_beta,
    pct_cdd_k,
    intercept,
    T_min,
    T_max,
    T_min_seg,
    T_max_seg,
    model_key,
    expected_coef_id,
    expected_x,
):
    coef_id, x = reduce_model(
        hdd_bp,
        hdd_beta,
        pct_hdd_k,
        cdd_bp,
        cdd_beta,
        pct_cdd_k,
        intercept,
        T_min,
        T_max,
        T_min_seg,
        T_max_seg,
        model_key,
    )

    assert coef_id == expected_coef_id
    np.allclose(x, expected_x)

def test_acf():
    # Test case 1: Test with a simple input array
    x = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([ 1. ,  0.4, -0.1, -0.4])
    assert np.allclose(acf(x), expected_output)

    # Test case 2: Test with a larger input array
    x = np.random.rand(100)
    expected_output = np.correlate(x - x.mean(), x - x.mean(), mode='full')[len(x)-1:]/(len(x)*x.var())
    assert np.allclose(acf(x), expected_output)

    # Test case 3: Test with a moving mean and standard deviation
    x = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([1.        , 0.79999999, 0.60000002, 0.39999998])
    assert np.allclose(acf(x, moving_mean_std=True), expected_output)

    # Test case 4: Test with a specific lag_n
    x = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([1.        , 0.79999999])
    assert np.allclose(acf(x, lag_n=1), expected_output)

class TestOptimizeResult:
    @pytest.fixture
    def optimize_result(self):
        # create an instance of OptimizeResult for testing
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        coef_id = ["hdd_bp", "hdd_beta", "hdd_k", "cdd_bp", "cdd_beta", "cdd_k", "intercept"]
        T = np.array([1, 2, 3, 4, 5])
        model = np.array([1, 2, 3, 4, 5])
        resid = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        settings = Settings()
        jac = None
        success = True
        message = "Optimization terminated successfully."
        nfev = 10
        time_elapsed = 1.0
        return OptimizedResult(x, coef_id, T, model, resid, weight, settings, jac, success, message, nfev, time_elapsed)

    def test_named_coeffs(self, optimize_result):
        # test that named_coeffs is an instance of ModelCoefficients
        assert isinstance(optimize_result.named_coeffs, ModelCoefficients)

    def test_prediction_uncertainty(self, optimize_result):
        # test that _prediction_uncertainty sets f_unc correctly
        optimize_result._prediction_uncertainty()
        assert optimize_result.f_unc == pytest.approx(0.1732050808)

    def test_set_model_key(self, optimize_result):
        # test that _set_model_key sets model_key and model_name correctly
        optimize_result._set_model_key()
        assert optimize_result.model_key == "hdd_tidd_cdd_smooth"
        assert optimize_result.model_name == "hdd_tidd_cdd_smooth"

    def test_refine_model(self, optimize_result):
        # test that _refine_model sets coef_id and x correctly
        optimize_result._refine_model()
        assert optimize_result.coef_id == ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]
        assert optimize_result.x == pytest.approx(np.array([1, 2, 4, 5, 7]))

    def test_eval(self, optimize_result):
        # test that eval returns the correct values
        T = np.array([1, 2, 3, 4, 5])
        model, f_unc, hdd_load, cdd_load = optimize_result.eval(T)
        assert model == pytest.approx(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert f_unc == pytest.approx(np.array([0.1732050808, 0.1732050808, 0.1732050808, 0.1732050808, 0.1732050808]))
        assert hdd_load == pytest.approx(np.array([1.0, 2.0, 0.0, 0.0, 0.0]))
        assert cdd_load == pytest.approx(np.array([0.0, 0.0, 3.0, 4.0, 5.0]))