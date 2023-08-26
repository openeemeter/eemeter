import numpy as np
import numba

from scipy.stats import linregress, theilslopes
from scipy.special import lambertw

from scipy.optimize import minimize_scalar

from eemeter.caltrack.daily.utilities.utils import OoM_numba

numba_cache = True
# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html


def get_intercept(y, alpha=2):
    """
    Calculates the intercept of a linear regression model.

    Parameters:
    -----------
    y : array-like
        Dependent variable.
    alpha : float, optional
        Significance level for the Theil-Sen estimator. Default is 2.

    Returns:
    --------
    intercept : float
        Intercept of the linear regression model.
    """
    
    if alpha == 2:
        intercept = np.mean(y)
    else:
        intercept = np.median(y)

    return intercept


def get_slope(x, y, x_bp, intercept, alpha=2):
    def slope_fcn_dec(x, y, x_bp, intercept, alpha):
        def slope_fcn(slope):  # TODO: This function could be numba'd
            model = slope * (x - x_bp) + intercept
            resid = y - model

            if alpha == 2:
                obj = np.sqrt(np.sum(resid**2))
            else:
                # obj = np.sum(np.abs(resid)) # MAE
                obj = np.sum(np.logaddexp(resid, -resid) - np.log(2))  # log cosh

            return obj

        return slope_fcn

    opt_fcn = slope_fcn_dec(x, y, x_bp, intercept, alpha)

    slope = minimize_scalar(opt_fcn, method="brent", tol=0.1).x

    return slope


def linear_fit(x, y, alpha):
    if alpha == 2:
        res = linregress(x, y)

        slope = res.slope
        intercept = res.intercept

    else:
        slope, intercept, _, _ = theilslopes(x, y, alpha=0.95)

    return slope, intercept


# smoothed curve will match unsmoothed at the perc_match% decay of the exp
# if pct_match == 1.0, then it will converge at - or + inf
def get_smooth_coeffs(hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k, min_pct_k=0.01):
    if (pct_hdd_k < min_pct_k) and (pct_cdd_k < min_pct_k):
        return np.array([hdd_bp, 0, cdd_bp, 0])

    pct_match = 1
    hdd_w = cdd_w = 0
    if pct_match < 1:
        hdd_w = lambertw((1 - pct_match) / np.exp(1)).real
        # cdd_w = lambertw(-(1 - pct_match)/np.exp(1)).real
        cdd_w = -hdd_w

    pct_k_sum = pct_hdd_k + pct_cdd_k
    if pct_k_sum > 1:
        pct_hdd_k /= pct_k_sum
        pct_cdd_k /= pct_k_sum

    # calculate the smoothing parameter as a percentage of the maximum allowed k
    hdd_k = pct_hdd_k * (cdd_bp - hdd_bp) / (1 - hdd_w)
    cdd_k = pct_cdd_k * (cdd_bp - hdd_bp) / (1 + cdd_w)

    # move breakpoints based on k
    hdd_bp = hdd_bp + hdd_k * (1 - hdd_w)
    cdd_bp = cdd_bp - cdd_k * (1 + cdd_w)

    return np.array([hdd_bp, hdd_k, cdd_bp, cdd_k])


@numba.jit(nopython=True, error_model="numpy", cache=numba_cache)
def fix_identical_bnds(bnds):
    for i in np.argwhere(bnds[:, 0] == bnds[:, 1]):
        bnds[i, :] = bnds[i, :] + np.array([-1.0, 1.0]) * 10 ** OoM_numba(
            bnds[i, 0], method="floor"
        )

    return bnds


def get_T_bnds(T, settings):
    n_min_seg = settings.segment_minimum_count

    T_min = np.min(T)
    T_max = np.max(T)
    T_min_seg = np.partition(T, n_min_seg)[n_min_seg]
    T_max_seg = np.partition(T, -n_min_seg)[-n_min_seg]

    return [T_min, T_max], [T_min_seg, T_max_seg]
