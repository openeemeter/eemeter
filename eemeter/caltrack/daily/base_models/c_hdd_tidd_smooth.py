import numpy as np
import numba

from eemeter.caltrack.daily.base_models.c_hdd_tidd import _c_hdd_tidd_x0, _c_hdd_tidd_x0_final
from eemeter.caltrack.daily.base_models.full_model import full_model
from eemeter.caltrack.daily.base_models.hdd_tidd_cdd_smooth import _hdd_tidd_cdd_smooth_weight

from eemeter.caltrack.daily.utilities.utils_base_model import fix_identical_bnds

from eemeter.caltrack.daily.objective_function import obj_fcn_decorator
from eemeter.caltrack.daily.optimize import Optimizer


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


def fit_c_hdd_tidd_smooth(T, obs, settings, opt_options, x0=None, bnds=None, initial_fit=False):
    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final 

    if x0 is None:
        x0 = _c_hdd_tidd_smooth_x0(T, obs, alpha, settings)
    else:
        x0 = _c_hdd_tidd_smooth_x0_final(T, obs, x0, alpha, settings)

    max_slope = np.abs(x0[1]) + 10**(np.log10(np.abs(x0[1])) + np.log10(settings.maximum_slope_OoM_scaler))

    if initial_fit:
        T_min = np.min(T)
        T_max = np.max(T)
    else:
        N_min = settings.segment_minimum_count

        T_min = np.partition(T, N_min)[N_min]
        T_max = np.partition(T, -N_min)[-N_min]

    c_hdd_bnds = [T_min, T_max]

    if initial_fit:
        c_hdd_beta_bnds = [-max_slope, max_slope]
    elif x0[1] < 0:
        c_hdd_beta_bnds = [-max_slope, 0]
    else:
        c_hdd_beta_bnds = [0, max_slope]

    c_hdd_k_bnds = [0, 1E3]                            # TODO: Need better estimation
    intercept_bnds = np.quantile(obs, [0.01, 0.99])
    bnds_0 = [c_hdd_bnds, c_hdd_beta_bnds, c_hdd_k_bnds, intercept_bnds]

    if bnds is None:
        bnds = bnds_0
    
    bnds = _c_hdd_tidd_smooth_update_bnds(bnds, bnds_0)

    coef_id = ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]
    model_fcn = _c_hdd_tidd_smooth
    weight_fcn = _c_hdd_tidd_smooth_weight
    TSS_fcn = None
    obj_fcn = lambda alpha, C: obj_fcn_decorator(model_fcn, weight_fcn, TSS_fcn, T, obs, settings, alpha, C, coef_id, initial_fit)

    res = Optimizer(obj_fcn, x0, bnds, coef_id, alpha, settings, opt_options).run()

    return res


# Model Functions
def _c_hdd_tidd_smooth_x0(T, obs, alpha, settings):
    [c_hdd_bp, c_hdd_beta, intercept] = _c_hdd_tidd_x0(T, obs, alpha, settings)
    c_hdd_k = 0.0

    return np.array([c_hdd_bp, c_hdd_beta, c_hdd_k, intercept])

def _c_hdd_tidd_smooth_x0_final(T, obs, x0, alpha, settings):
    c_hdd_bp, c_hdd_beta, c_hdd_k, intercept = x0
    x0_fit = [c_hdd_bp, c_hdd_beta, intercept]
    [c_hdd_bp, c_hdd_beta, intercept] = _c_hdd_tidd_x0_final(T, obs, x0_fit, alpha, settings)
    
    return np.array([c_hdd_bp, c_hdd_beta, c_hdd_k, intercept])


@numba.jit(nopython=True, error_model='numpy', cache=numba_cache) 
def set_full_model_coeffs(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept):
    hdd_bp = cdd_bp = c_hdd_bp

    if c_hdd_beta < 0:
        hdd_beta = -c_hdd_beta
        hdd_k = c_hdd_k
        cdd_beta = cdd_k = 0

    else:
        cdd_beta = c_hdd_beta
        cdd_k = c_hdd_k
        hdd_beta = hdd_k = 0

    return np.array([hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept])


def _c_hdd_tidd_smooth(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept, T_fit_bnds=np.array([]), T=np.array([])):
    x = set_full_model_coeffs(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept)

    return full_model(*x, T_fit_bnds, T)


def _c_hdd_tidd_smooth_update_bnds(new_bnds, bnds):
    new_bnds[0] = bnds[0]
    new_bnds[3] = bnds[3]

    new_bnds = np.sort(new_bnds, axis=1)
    new_bnds = fix_identical_bnds(new_bnds)

    if new_bnds[2, 0] < 0:
        new_bnds[2, 0] = 0

    return new_bnds


def _c_hdd_tidd_smooth_weight(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept, T, residual, sigma=3.0, quantile=0.25, alpha=2.0, min_weight=0.0):
    model_vars = set_full_model_coeffs(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept)
    
    return _hdd_tidd_cdd_smooth_weight(*model_vars, T, residual, sigma, quantile, alpha, min_weight)