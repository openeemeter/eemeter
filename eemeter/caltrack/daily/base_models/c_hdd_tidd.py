import numpy as np
from math import isclose
import numba
import nlopt

from eemeter.caltrack.daily.base_models.full_model import full_model
from eemeter.caltrack.daily.base_models.hdd_tidd_cdd_smooth import (
    _hdd_tidd_cdd_smooth_weight,
)

from eemeter.caltrack.daily.utilities.adaptive_loss import adaptive_weights

from eemeter.caltrack.daily.utilities.base_model import (
    linear_fit,
    get_slope,
    get_intercept,
    get_T_bnds,
)
from eemeter.caltrack.daily.utilities.base_model import fix_identical_bnds

from eemeter.caltrack.daily.objective_function import obj_fcn_decorator
from eemeter.caltrack.daily.optimize import Optimizer, nlopt_algorithms


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True

# how close can the temperature be to the segment_minimum_temp and still be considered a slope only?
rtol = 1e-5


"""TODO
check for differences in beta bounds between this and smoothed model. merge both first rather than
adding the ModelCoefficients here, as the current implementation updates x0 and is a bit cumbersome to convert
"""
def fit_c_hdd_tidd(
    T, obs, settings, opt_options, x0=None, bnds=None, initial_fit=False
):
    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final

    if x0 is None:
        x0 = _c_hdd_tidd_x0(T, obs, alpha, settings)
    else:
        x0 = _c_hdd_tidd_x0_final(T, obs, x0, alpha, settings)

    max_slope = np.abs(x0[1]) + 10 ** (
        np.log10(np.abs(x0[1])) + np.log10(settings.maximum_slope_OoM_scaler)
    )

    # standard T_min and T_max for initial guess
    [T_min, T_max], [T_min_seg, T_max_seg] = get_T_bnds(T, settings)

    # set bounds and initial guesses for complex cases
    if not initial_fit:
        if x0[0] <= T_min_seg or isclose(x0[0], T_min_seg, rel_tol=rtol):
            x0[2] -= x0[1] * T_min
            x0[0] = T_min
            T_max = T_min

        elif x0[0] >= T_max_seg or isclose(x0[0], T_max_seg, rel_tol=rtol):
            x0[2] -= x0[1] * T_max
            x0[0] = T_max
            T_min = T_max

        else:
            T_min = T_min_seg
            T_max = T_max_seg

    c_hdd_bnds = [T_min, T_max]
    if initial_fit:
        c_hdd_beta_bnds = [-max_slope, max_slope]
    elif x0[1] < 0:
        c_hdd_beta_bnds = [-max_slope, 0]
    else:
        c_hdd_beta_bnds = [0, max_slope]

    intercept_bnds = np.quantile(obs, [0.01, 0.99])
    bnds_0 = [c_hdd_bnds, c_hdd_beta_bnds, intercept_bnds]

    if bnds is None:
        bnds = bnds_0

    bnds = _c_hdd_tidd_update_bnds(bnds, bnds_0)

    # override bnds correction. Normally great, but not here
    if c_hdd_bnds[0] == c_hdd_bnds[1]:
        bnds[0, :] = c_hdd_bnds

    coef_id = ["c_hdd_bp", "c_hdd_beta", "intercept"]
    model_fcn = _c_hdd_tidd
    weight_fcn = _c_hdd_tidd_weight
    TSS_fcn = _c_hdd_tidd_total_sum_of_squares
    obj_fcn = lambda alpha, C: obj_fcn_decorator(
        model_fcn, weight_fcn, TSS_fcn, T, obs, settings, alpha, C, coef_id, initial_fit
    )

    res = Optimizer(obj_fcn, x0, bnds, coef_id, alpha, settings, opt_options).run()

    return res


# Model Functions
def _c_hdd_tidd_x0(T, obs, alpha, settings):
    min_T_idx = settings.segment_minimum_count

    # c_hdd_bp = initial_guess_bp_1(T, obs, s=2, int_method="trapezoid")
    c_hdd_bp = _c_hdd_tidd_bp0(T, obs, alpha, settings)
    c_hdd_bp = np.clip([c_hdd_bp], T[min_T_idx - 1], T[-min_T_idx])[0]

    idx_hdd = np.argwhere(T <= c_hdd_bp).flatten()
    idx_cdd = np.argwhere(T >= c_hdd_bp).flatten()

    hdd_beta, _ = linear_fit(obs[idx_hdd], T[idx_hdd], alpha)
    if hdd_beta > 0:
        hdd_beta = 0

    cdd_beta, _ = linear_fit(obs[idx_cdd], T[idx_cdd], alpha)
    if cdd_beta < 0:
        cdd_beta = 0

    if -hdd_beta >= cdd_beta:
        c_hdd_beta = hdd_beta
        intercept = np.median(obs[idx_cdd])

    else:
        c_hdd_beta = cdd_beta
        intercept = np.median(obs[idx_hdd])

    return np.array([c_hdd_bp, c_hdd_beta, intercept])


def _c_hdd_tidd_x0_final(T, obs, x0, alpha, settings):
    min_T_idx = settings.segment_minimum_count
    c_hdd_bp, c_hdd_beta, intercept = x0

    idx_hdd = np.argwhere(T <= c_hdd_bp).flatten()
    idx_cdd = np.argwhere(T >= c_hdd_bp).flatten()

    if (c_hdd_beta < 0) and (len(idx_hdd) >= min_T_idx):  # hdd
        c_hdd_beta = get_slope(T[idx_hdd], obs[idx_hdd], c_hdd_bp, intercept, alpha)

    elif (c_hdd_beta >= 0) and (len(idx_cdd) >= min_T_idx):  # cdd
        c_hdd_beta = get_slope(T[idx_cdd], obs[idx_cdd], c_hdd_bp, intercept, alpha)

    return np.array([c_hdd_bp, c_hdd_beta, intercept])


def _c_hdd_tidd_bp0(T, obs, alpha, settings, min_weight=0.0):
    min_T_idx = settings.segment_minimum_count

    idx_sorted = np.argsort(T).flatten()
    T = T[idx_sorted]
    obs = obs[idx_sorted]

    def bp_obj_fcn_dec(T, obs):
        def bp_obj_fcn(x, grad=[]):
            [c_hdd_bp] = x

            idx_hdd = np.argwhere(T <= c_hdd_bp).flatten()
            idx_cdd = np.argwhere(T >= c_hdd_bp).flatten()

            hdd_beta, _ = linear_fit(obs[idx_hdd], T[idx_hdd], alpha)
            if hdd_beta > 0:
                hdd_beta = 0

            cdd_beta, _ = linear_fit(obs[idx_cdd], T[idx_cdd], alpha)
            if cdd_beta < 0:
                cdd_beta = 0

            if -hdd_beta >= cdd_beta:
                c_hdd_beta = hdd_beta
                intercept = get_intercept(obs[idx_cdd], alpha)

            else:
                c_hdd_beta = cdd_beta
                intercept = get_intercept(obs[idx_hdd], alpha)

            model = _c_hdd_tidd(c_hdd_bp, c_hdd_beta, intercept, T=T)

            resid = model - obs
            weight, _ = adaptive_weights(resid, alpha=alpha, min_weight=min_weight)

            loss = np.sum(weight * (resid) ** 2)

            return loss

        return bp_obj_fcn

    algorithm = nlopt_algorithms[settings.initial_guess_algorithm_choice]
    # algorithm = nlopt.GN_DIRECT

    obj_fcn = bp_obj_fcn_dec(T, obs)

    T_min = T[min_T_idx - 1]
    T_max = T[-min_T_idx]
    T_range = T_max - T_min

    x0 = np.array([T_range * 0.5]) + T_min
    bnds = np.array([[T_min, T_max]]).T

    opt = nlopt.opt(algorithm, int(len(x0)))
    opt.set_min_objective(obj_fcn)

    opt.set_initial_step([T_range * 0.25])
    opt.set_maxeval(100)
    opt.set_xtol_rel(1e-3)
    opt.set_xtol_abs(0.5)
    opt.set_lower_bounds(bnds[0])
    opt.set_upper_bounds(bnds[1])

    x_opt = opt.optimize(x0)  # optimize!

    return x_opt[0]


@numba.jit(nopython=True, error_model="numpy", cache=numba_cache)
def set_full_model_coeffs(c_hdd_bp, c_hdd_beta, intercept):
    hdd_bp = cdd_bp = c_hdd_bp

    if c_hdd_beta < 0:
        hdd_beta = -c_hdd_beta
        cdd_beta = cdd_k = hdd_k = 0

    else:
        cdd_beta = c_hdd_beta
        hdd_beta = hdd_k = cdd_k = 0

    return np.array([hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept])


def _c_hdd_tidd(
    c_hdd_bp, c_hdd_beta, intercept, T_fit_bnds=np.array([]), T=np.array([])
):
    model_vars = set_full_model_coeffs(c_hdd_bp, c_hdd_beta, intercept)

    return full_model(*model_vars, T_fit_bnds, T)


def _c_hdd_tidd_total_sum_of_squares(c_hdd_bp, c_hdd_beta, intercept, T, obs):
    idx_bp = np.argmin(np.abs(T - c_hdd_bp))

    TSS = []
    for observed in [obs[:idx_bp], obs[idx_bp:]]:
        if len(observed) == 0:
            continue

        TSS.append(np.sum((observed - np.mean(observed)) ** 2))

    TSS = np.sum(TSS)

    return TSS


def _c_hdd_tidd_update_bnds(new_bnds, bnds):
    new_bnds[0] = bnds[0]
    new_bnds[2] = bnds[2]

    new_bnds = np.sort(new_bnds, axis=1)
    new_bnds = fix_identical_bnds(new_bnds)

    return new_bnds


def _c_hdd_tidd_weight(
    c_hdd_bp,
    c_hdd_beta,
    intercept,
    T,
    residual,
    sigma=3.0,
    quantile=0.25,
    alpha=2.0,
    min_weight=0.0,
):
    model_vars = set_full_model_coeffs(c_hdd_bp, c_hdd_beta, intercept)

    return _hdd_tidd_cdd_smooth_weight(
        *model_vars, T, residual, sigma, quantile, alpha, min_weight
    )
