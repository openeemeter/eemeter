import numpy as np
import nlopt

from eemeter.caltrack.daily.base_models.full_model import full_model_weight

from eemeter.caltrack.daily.base_models.full_model_import_finder import full_model
from eemeter.caltrack.daily.base_models.full_model import get_full_model_x

from eemeter.caltrack.daily.utilities.adaptive_loss import adaptive_weights, get_C
from eemeter.caltrack.daily.utilities.adaptive_loss import remove_outliers

from eemeter.caltrack.daily.utilities.base_model import get_slope, get_intercept
from eemeter.caltrack.daily.utilities.base_model import fix_identical_bnds
from eemeter.caltrack.daily.utilities.base_model import get_smooth_coeffs

from eemeter.caltrack.daily.objective_function import obj_fcn_decorator
from eemeter.caltrack.daily.optimize import Optimizer, nlopt_algorithms

from timeit import default_timer as timer
from eemeter.caltrack.daily.utilities.utils import ModelCoefficients, ModelType
from typing import Optional


# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


# TODO: Might be able to make fitting faster by optimizing bp's and intercept and beta/pct_k inside
def fit_hdd_tidd_cdd_smooth(
    T,
    obs,
    settings,
    opt_options,
    x0: Optional[ModelCoefficients] = None,
    bnds=None,
    initial_fit=False,
):
    assert x0 is None or x0.model_type is ModelType.HDD_TIDD_CDD_SMOOTH

    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final

    if x0 is None:
        x0 = _hdd_tidd_cdd_smooth_x0(T, obs, alpha, settings)

    max_slope = np.max([x0.hdd_beta, x0.cdd_beta])
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
    obj_fcn = obj_fcn_decorator(
        model_fcn, weight_fcn, TSS_fcn, T, obs, settings, alpha, coef_id, initial_fit
    )

    res = Optimizer(
        obj_fcn, x0.to_np_array(), bnds, coef_id, settings, opt_options
    ).run()

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


def _hdd_tidd_cdd_smooth_x0(T, obs, alpha, settings, min_weight=0.0):
    min_T_idx = settings.segment_minimum_count
    lasso_a = settings.regularization_alpha

    idx_sorted = np.argsort(T).flatten()
    T = T[idx_sorted]
    obs = obs[idx_sorted]

    N = len(obs)

    T_fit_bnds = np.array([T[0], T[-1]])

    def bp_obj_fcn_dec(T, obs, min_T_idx):
        def lasso_penalty(X, wRMSE):
            X_lasso = np.array(X).copy()

            T_range = T_fit_bnds[1] - T_fit_bnds[0]

            X_lasso = np.array(
                [np.min(np.abs(X[idx] - T_fit_bnds)) for idx in range(len(X))]
            )
            X_lasso += (X[1] - X[0]) / 2
            X_lasso *= wRMSE / T_range

            return lasso_a * np.linalg.norm(X_lasso, 1)

        def bp_obj_fcn(x, grad=[], optimize_flag=True):
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

            if alpha == 2:
                resid_mean = np.mean(resid)
                resid -= resid_mean
                intercept += resid_mean
            else:
                resid_median = np.median(resid)
                resid -= resid_median
                intercept += resid_median

            weight, _, _ = adaptive_weights(
                resid, alpha=alpha, sigma=2.698, quantile=0.25, min_weight=min_weight
            )

            loss = np.sum(weight * (resid) ** 2)
            loss += lasso_penalty(x, np.sqrt(loss / N))

            if optimize_flag:
                return loss

            return np.array(
                [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]
            )

        return bp_obj_fcn

    algorithm = nlopt_algorithms[settings.initial_guess_algorithm_choice]
    obj_fcn = bp_obj_fcn_dec(T, obs, min_T_idx)

    T_min = T[min_T_idx - 1]
    T_max = T[-min_T_idx]
    T_range = T_max - T_min

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

    x_opt = opt.optimize(x0)  # optimize!

    x0 = obj_fcn(x_opt, optimize_flag=False)

    return ModelCoefficients(
        model_type=ModelType.HDD_TIDD_CDD_SMOOTH,
        hdd_bp=x0[0],
        hdd_beta=x0[1],
        hdd_k=x0[2],
        cdd_bp=x0[3],
        cdd_beta=x0[4],
        cdd_k=x0[5],
        intercept=x0[6],
    )


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
    model_vars = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

    return full_model_weight(
        *model_vars, T, residual, sigma, quantile, alpha, min_weight
    )
