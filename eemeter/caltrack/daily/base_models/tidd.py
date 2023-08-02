import numpy as np
import numba

from eemeter.caltrack.daily.base_models.full_model import full_model, full_model_weight

from eemeter.caltrack.daily.utilities.base_model import fix_identical_bnds

from eemeter.caltrack.daily.objective_function import obj_fcn_decorator

from eemeter.caltrack.daily.optimize import Optimizer

from eemeter.caltrack.daily.utilities.utils import ModelCoefficients, ModelType
from typing import Optional

# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


def fit_tidd(
    T,
    obs,
    settings,
    opt_options,
    x0: Optional[ModelCoefficients] = None,
    bnds=None,
    initial_fit=False,
):
    if x0 is None:
        x0 = _tidd_x0(T, obs)

    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final

    intercept_bnds = np.quantile(obs, [0.01, 0.99])
    bnds_0 = np.array([intercept_bnds])

    if bnds is None:
        bnds = bnds_0

    bnds = _tidd_update_bnds(bnds, bnds_0)

    coef_id = ["intercept"]
    model_fcn = _tidd
    weight_fcn = _tidd_weight
    TSS_fcn = _tidd_total_sum_of_squares
    obj_fcn = obj_fcn_decorator(
        model_fcn, weight_fcn, TSS_fcn, T, obs, settings, alpha, coef_id, initial_fit
    )

    res = Optimizer(
        obj_fcn, x0.to_np_array(), bnds, coef_id, settings, opt_options
    ).run()

    return res


# Model Functions
def _tidd_x0(T, obs):
    intercept = np.median(obs)
    return ModelCoefficients(model_type=ModelType.TIDD, intercept=intercept)


@numba.jit(nopython=True, error_model="numpy", cache=numba_cache)
def set_full_model_coeffs(intercept):
    hdd_bp = hdd_beta = hdd_k = cdd_bp = cdd_beta = cdd_k = 0

    return np.array([hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept])


def _tidd(intercept, T_fit_bnds=np.array([]), T=np.array([])):
    model_vars = set_full_model_coeffs(intercept)

    return full_model(*model_vars, T_fit_bnds, T)


def _tidd_total_sum_of_squares(intercept, T, obs):
    TSS = np.sum((obs - np.mean(obs)) ** 2)

    return TSS


def _tidd_update_bnds(new_bnds, bnds):
    new_bnds = bnds

    new_bnds = np.sort(new_bnds, axis=1)
    new_bnds = fix_identical_bnds(new_bnds)

    return new_bnds


def _tidd_weight(
    intercept, T, residual, sigma=3.0, quantile=0.25, alpha=2.0, min_weight=0.0
):
    model_vars = set_full_model_coeffs(intercept)

    return full_model_weight(
        *model_vars, T, residual, sigma, quantile, alpha, min_weight
    )
