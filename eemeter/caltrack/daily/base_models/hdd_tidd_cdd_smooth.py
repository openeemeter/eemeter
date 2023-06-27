import numpy as np
import nlopt

from scipy.stats import linregress

from eemeter.caltrack.daily.base_models.full_model import full_model, get_full_model_x

from eemeter.caltrack.daily.utilities.adaptive_loss import adaptive_weights, get_C
from eemeter.caltrack.daily.utilities.adaptive_loss import remove_outliers

from eemeter.caltrack.daily.utilities.base_model import get_slope, get_intercept
from eemeter.caltrack.daily.utilities.base_model import fix_identical_bnds
from eemeter.caltrack.daily.utilities.base_model import (
    get_smooth_coeffs,
    get_T_bnds,
)

from eemeter.caltrack.daily.objective_function import obj_fcn_decorator
from eemeter.caltrack.daily.optimize import Optimizer, nlopt_algorithms


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


# TODO: Might be able to make fitting faster by optimizing bp's and intercept and beta/pct_k inside
def fit_hdd_tidd_cdd_smooth(
    T, obs, settings, opt_options, x0=None, bnds=None, initial_fit=False
):
    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final

    if x0 is None:
        x0 = _hdd_tidd_cdd_smooth_x0(T, obs, alpha, settings)

    # max_slope = np.max([x0[1], x0[4]])*settings.maximum_slope_multiplier # TODO: Need better estimation
    max_slope = np.max([x0[1], x0[4]])
    max_slope += 10 ** (
        np.log10(np.abs(max_slope)) + np.log10(settings.maximum_slope_OoM_scaler)
    )

    if initial_fit:
        T_min = np.min(T)
        T_max = np.max(T)
    else:
        N_min = settings.segment_minimum_count

        T_min = np.partition(T, N_min)[N_min]
        T_max = np.partition(T, -N_min)[-N_min]

    c_hdd_bnds = [T_min, T_max]
    c_hdd_beta_bnds = [0, np.abs(max_slope)]
    c_hdd_k_bnds = [0, 1]
    intercept_bnds = np.quantile(obs, [0.01, 0.99])
    bnds_0 = [
        c_hdd_bnds,
        c_hdd_beta_bnds,
        c_hdd_k_bnds,
        c_hdd_bnds,
        c_hdd_beta_bnds,
        c_hdd_k_bnds,
        intercept_bnds,
    ]

    if bnds is None:
        bnds = bnds_0

    bnds = _hdd_tidd_cdd_smooth_update_bnds(bnds, bnds_0)

    coef_id = [
        "hdd_bp",
        "hdd_beta",
        "hdd_k",
        "cdd_bp",
        "cdd_beta",
        "cdd_k",
        "intercept",
    ]
    model_fcn = evaluate_hdd_tidd_cdd_smooth
    weight_fcn = _hdd_tidd_cdd_smooth_weight
    TSS_fcn = None
    obj_fcn = lambda alpha, C: obj_fcn_decorator(
        model_fcn, weight_fcn, TSS_fcn, T, obs, settings, alpha, C, coef_id, initial_fit
    )

    res = Optimizer(obj_fcn, x0, bnds, coef_id, alpha, settings, opt_options).run()

    return res


# Model Functions
def _hdd_tidd_cdd_smooth(*args):
    return full_model(*args)


def evaluate_hdd_tidd_cdd_smooth(
    hdd_bp,
    hdd_beta,
    hdd_k,
    cdd_bp,
    cdd_beta,
    cdd_k,
    intercept,
    T_fit_bnds,
    T,
    pct_k=True,
):
    if pct_k:
        [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_smooth_coeffs(hdd_bp, hdd_k, cdd_bp, cdd_k)

    return _hdd_tidd_cdd_smooth(
        hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T
    )


def _hdd_tidd_cdd_smooth_x0(T, obs, alpha, settings, test_c_hdd=False):
    min_T_idx = settings.segment_minimum_count

    # calculate x0 for hdd_tidd_cdd_smooth
    [hdd_bp, cdd_bp] = _hdd_tidd_cdd_bp0(T, obs, alpha, settings)
    hdd_beta, cdd_beta, intercept = estimate_betas_and_intercept(
        T, obs, hdd_bp, cdd_bp, min_T_idx, alpha
    )
    hdd_k = 0.0
    cdd_k = 0.0

    x0_hdd_tidd_cdd = np.array(
        [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]
    )

    if test_c_hdd:  # Much slower including this. Maybe remove entirely
        # calculate x0 for c_hdd
        [T_min, T_max], [T_min_seg, T_max_seg] = get_T_bnds(T, settings)

        slope, intercept, _, _, _ = linregress(T, obs)

        # convert coefficients from standard linear to offset
        if slope < 0:
            x0_c_hdd = [T_max, slope, intercept + slope * T_max]
        else:
            x0_c_hdd = [T_min, slope, intercept + slope * T_min]

        x0_c_hdd = get_full_model_x(
            "c_hdd_tidd", x0_c_hdd, T_min, T_max, T_min_seg, T_max_seg
        )

        # get loss value
        coef_id = [
            "hdd_bp",
            "hdd_beta",
            "hdd_k",
            "cdd_bp",
            "cdd_beta",
            "cdd_k",
            "intercept",
        ]
        model_fcn = evaluate_hdd_tidd_cdd_smooth
        weight_fcn = _hdd_tidd_cdd_smooth_weight
        TSS_fcn = None
        # alpha = settings.alpha_final
        C = None
        obj_fcn = obj_fcn_decorator(
            model_fcn, weight_fcn, TSS_fcn, T, obs, settings, alpha, C, coef_id
        )

        if obj_fcn(x0_c_hdd) < obj_fcn(x0_hdd_tidd_cdd):
            x0 = x0_c_hdd
        else:
            x0 = x0_hdd_tidd_cdd

    else:
        x0 = x0_hdd_tidd_cdd

    return np.array(x0)


def _hdd_tidd_cdd_bp0(T, obs, alpha, settings, min_weight=0.0, test_num_bp=[1, 2]):
    min_T_idx = settings.segment_minimum_count

    idx_sorted = np.argsort(T).flatten()
    T = T[idx_sorted]
    obs = obs[idx_sorted]

    T_fit_bnds = np.array([T[0], T[-1]])

    def bp_obj_fcn_dec(T, obs, min_T_idx):
        def bp_obj_fcn(x, grad=[]):
            if len(x) == 1:
                hdd_bp = cdd_bp = x[0]
            else:
                [hdd_bp, cdd_bp] = np.sort(x)

            hdd_beta, cdd_beta, intercept = estimate_betas_and_intercept(
                T, obs, hdd_bp, cdd_bp, min_T_idx, alpha
            )
            hdd_k = cdd_k = 0

            model = _hdd_tidd_cdd_smooth(
                hdd_bp,
                hdd_beta,
                hdd_k,
                cdd_bp,
                cdd_beta,
                cdd_k,
                intercept,
                T_fit_bnds,
                T,
            )

            resid = model - obs
            weight, _ = adaptive_weights(resid, alpha=alpha, min_weight=min_weight)

            loss = np.sum(weight * (resid) ** 2)

            return loss

        return bp_obj_fcn

    algorithm = nlopt_algorithms[settings.initial_guess_algorithm_choice]
    obj_fcn = bp_obj_fcn_dec(T, obs, min_T_idx)

    T_min = T[min_T_idx - 1]
    T_max = T[-min_T_idx]
    T_range = T_max - T_min

    # 1 breakpoint guess
    if 1 in test_num_bp:
        # algorithm = nlopt.GN_DIRECT
        x0 = np.array([T_range * 0.5]) + T_min
        bnds = np.array([[T_min, T_max]]).T

        opt = nlopt.opt(algorithm, int(len(x0)))
        opt.set_min_objective(obj_fcn)

        opt.set_initial_step([T_range * 0.15])
        opt.set_maxeval(100)
        opt.set_xtol_rel(1e-3)
        opt.set_xtol_abs(0.5)
        opt.set_lower_bounds(bnds[0])
        opt.set_upper_bounds(bnds[1])

        x_opt_1 = opt.optimize(x0)  # optimize!
        x_opt_1 = np.array([x_opt_1[0], x_opt_1[0]])
        f_opt_1 = opt.last_optimum_value()

    else:
        x_opt_1 = np.array([np.nan, np.nan])
        f_opt_1 = np.inf

    # 2 breakpoints guess
    if 2 in test_num_bp:
        x0 = np.array([T_range * 0.10, T_range * 0.90]) + T_min
        bnds = np.array([[T_min, T_max], [T_min, T_max]]).T

        opt = nlopt.opt(algorithm, int(len(x0)))
        opt.set_min_objective(obj_fcn)

        opt.set_initial_step([T_range * 0.10, -T_range * 0.10])
        opt.set_maxeval(200)
        opt.set_xtol_rel(1e-3)
        opt.set_xtol_abs(0.5)
        opt.set_lower_bounds(bnds[0])
        opt.set_upper_bounds(bnds[1])

        x_opt_2 = opt.optimize(x0)  # optimize!
        x_opt_2 = np.sort(x_opt_2)
        f_opt_2 = opt.last_optimum_value()

    else:
        x_opt_2 = np.array([np.nan, np.nan])
        f_opt_2 = np.inf

    if f_opt_1 <= f_opt_2:
        x_opt = x_opt_1
    else:
        x_opt = x_opt_2

    x_opt[0] = np.clip([x_opt[0]], T[min_T_idx - 1], T[-min_T_idx])[0]
    x_opt[1] = np.clip([x_opt[1]], T[min_T_idx - 1], T[-min_T_idx])[0]

    return x_opt


def estimate_betas_and_intercept(T, obs, hdd_bp, cdd_bp, min_T_idx, alpha):
    idx_hdd = np.argwhere(T < hdd_bp).flatten()
    idx_tidd = np.argwhere((hdd_bp <= T) & (T <= cdd_bp)).flatten()
    idx_cdd = np.argwhere(cdd_bp < T).flatten()

    if len(idx_tidd) > 0:
        intercept = get_intercept(obs[idx_tidd], alpha)
    elif (idx_cdd[min_T_idx - 1] - idx_hdd[-min_T_idx]) > 0:
        intercept = get_intercept(
            obs[idx_hdd[-min_T_idx] : idx_cdd[min_T_idx - 1]], alpha
        )
    else:
        intercept = np.quantile(obs, 0.20)

    hdd_beta = get_slope(T[idx_hdd], obs[idx_hdd], hdd_bp, intercept, alpha)
    if hdd_beta > 0:
        hdd_beta = 0
    else:
        hdd_beta *= -1

    cdd_beta = get_slope(T[idx_cdd], obs[idx_cdd], cdd_bp, intercept, alpha)
    if cdd_beta < 0:
        cdd_beta = 0

    return hdd_beta, cdd_beta, intercept


def _hdd_tidd_cdd_smooth_update_bnds(new_bnds, bnds):
    new_bnds[0] = bnds[0]
    new_bnds[3] = bnds[3]
    new_bnds[6] = bnds[6]

    new_bnds = np.sort(new_bnds, axis=1)
    new_bnds = fix_identical_bnds(new_bnds)

    for i in [1, 2, 4, 5]:
        if new_bnds[i][0] < 0:
            new_bnds[i][0] = 0

    return new_bnds


def _hdd_tidd_cdd_smooth_weight(
    hdd_bp,
    hdd_beta,
    hdd_k,
    cdd_bp,
    cdd_beta,
    cdd_k,
    intercept,
    T,
    residual,
    sigma=3.0,
    quantile=0.25,
    alpha=2.0,
    min_weight=0.0,
):
    if hdd_bp > cdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp

    if (hdd_beta == 0) and (cdd_beta == 0):  # intercept only
        resid_all = [residual]

    elif (cdd_bp >= T[-1]) or (hdd_bp <= T[0]):  # hdd or cdd only
        resid_all = [residual]

    elif hdd_beta == 0:
        idx_cdd_bp = np.argmin(np.abs(T - cdd_bp))

        resid_all = [residual[:idx_cdd_bp], residual[idx_cdd_bp:]]

    elif cdd_beta == 0:
        idx_hdd_bp = np.argmin(np.abs(T - hdd_bp))

        resid_all = [residual[:idx_hdd_bp], residual[idx_hdd_bp:]]

    else:
        idx_hdd_bp = np.argmin(np.abs(T - hdd_bp))
        idx_cdd_bp = np.argmin(np.abs(T - cdd_bp))

        if hdd_bp == cdd_bp:
            resid_all = [residual[:idx_hdd_bp], residual[idx_cdd_bp:]]

        else:
            resid_all = [
                residual[:idx_hdd_bp],
                residual[idx_hdd_bp:idx_cdd_bp],
                residual[idx_cdd_bp:],
            ]

    weight = []
    C = []
    a = []
    for resid in resid_all:
        if len(resid) == 0:
            continue

        elif len(resid) < 3:
            weight.append(np.ones_like(resid))
            C.append(np.ones_like(resid))
            a.append(np.ones_like(resid) * 2.0)

            continue

        resid_no_outlier, _ = remove_outliers(
            resid, sigma_threshold=sigma, quantile=0.25
        )

        # mu = np.median(np.abs(resid_no_outlier))
        mu = np.median(resid_no_outlier)

        _C = get_C(resid, mu, sigma, quantile)
        resid_norm = (resid - mu) / _C
        _weight, _a = adaptive_weights(resid_norm, alpha=alpha, min_weight=min_weight)

        weight.append(_weight)
        C.append(np.ones_like(resid) * _C)
        a.append(np.ones_like(resid) * _a)

    weight_out = np.hstack(weight)
    C_out = np.hstack(weight)
    a_out = np.hstack(a)

    return weight_out, C_out, a_out
