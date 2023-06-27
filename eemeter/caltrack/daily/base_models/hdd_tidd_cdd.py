import numpy as np
import numba

from eemeter.caltrack.daily.base_models.full_model import full_model
from eemeter.caltrack.daily.base_models.hdd_tidd_cdd_smooth import _hdd_tidd_cdd_smooth_x0
from eemeter.caltrack.daily.base_models.hdd_tidd_cdd_smooth import _hdd_tidd_cdd_smooth_weight

from eemeter.caltrack.daily.utilities.utils_base_model import fix_identical_bnds

from eemeter.caltrack.daily.objective_function import obj_fcn_decorator
from eemeter.caltrack.daily.optimize import Optimizer


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


def fit_hdd_tidd_cdd(T, obs, settings, opt_options, x0=None, bnds=None, initial_fit=False):
    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final 

    if x0 is None:
        x0 = _hdd_tidd_cdd_x0(T, obs, alpha, settings)

    max_slope = np.max([x0[1], x0[3]])
    max_slope += 10**(np.log10(np.abs(max_slope)) + np.log10(settings.maximum_slope_OoM_scaler))

    if initial_fit:
        T_min = np.min(T)
        T_max = np.max(T)
    else:
        N_min = settings.segment_minimum_count

        T_min = np.partition(T, N_min)[N_min]
        T_max = np.partition(T, -N_min)[-N_min]

    c_hdd_bnds = [T_min, T_max]
    c_hdd_beta_bnds = [0, np.abs(max_slope)]
    intercept_bnds = np.quantile(obs, [0.01, 0.99])
    bnds_0 = [c_hdd_bnds, c_hdd_beta_bnds, 
              c_hdd_bnds, c_hdd_beta_bnds, 
              intercept_bnds]

    if bnds is None:
        bnds = bnds_0
    
    bnds = _hdd_tidd_cdd_update_bnds(bnds, bnds_0)

    coef_id = ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]
    model_fcn = _hdd_tidd_cdd
    weight_fcn = _hdd_tidd_cdd_weight
    TSS_fcn = _hdd_tidd_cdd_total_sum_of_squares
    obj_fcn = lambda alpha, C: obj_fcn_decorator(model_fcn, weight_fcn, TSS_fcn, T, obs, settings, alpha, C, coef_id, initial_fit)

    res = Optimizer(obj_fcn, x0, bnds, coef_id, alpha, settings, opt_options).run()

    return res


# Model Functions
@numba.jit(nopython=True, error_model='numpy', cache=numba_cache)
def _hdd_tidd_cdd(hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept, T_fit_bnds=np.array([]), T=np.array([])):
    hdd_k = cdd_k = 0

    return full_model(hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T)


def _hdd_tidd_cdd_x0(T, obs, alpha, settings):
    x0 = _hdd_tidd_cdd_smooth_x0(T, obs, alpha, settings)

    [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept] = x0
    x0 = [hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept] 

    return np.array(x0)


def _hdd_tidd_cdd_total_sum_of_squares(hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept, T, obs):
    if hdd_bp > cdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp

    idx_hdd_bp = np.argmin(np.abs(T - hdd_bp))
    idx_cdd_bp = np.argmin(np.abs(T - cdd_bp))

    TSS = []
    for observed in [obs[:idx_hdd_bp], obs[idx_hdd_bp:idx_cdd_bp], obs[idx_cdd_bp:]]:
        if len(observed) == 0:
            continue

        TSS.append(np.sum((observed - np.mean(observed))**2))

    TSS = np.sum(TSS)

    return TSS


def _hdd_tidd_cdd_update_bnds(new_bnds, bnds):
    new_bnds[0] = bnds[0]
    new_bnds[2] = bnds[2]
    new_bnds[4] = bnds[4]

    new_bnds = np.sort(new_bnds, axis=1)
    new_bnds = fix_identical_bnds(new_bnds)

    for i in [1, 3]:
        if new_bnds[i, 0] < 0:
            new_bnds[i, 0] = 0

    return new_bnds


def _hdd_tidd_cdd_weight(hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept, T, residual, sigma=3.0, quantile=0.25, alpha=2.0, min_weight=0.0):
    hdd_k = cdd_k = 0
    model_vars = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]
    
    return _hdd_tidd_cdd_smooth_weight(*model_vars, T, residual, sigma, quantile, alpha, min_weight)