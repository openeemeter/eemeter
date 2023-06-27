import numpy as np
import numba

from eemeter.caltrack.daily.utilities.utils import ln_min_pos_system_value, ln_max_pos_system_value

# To compile ahead of time: https://numba.readthedocs.io/en/stable/user/pycc.html
numba_cache = True


@numba.jit(nopython=True, error_model='numpy', cache=numba_cache)
def full_model(hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds=np.array([]), T=np.array([])):
    # if all variables are zero, return tidd model
    if (hdd_beta == 0) and (cdd_beta == 0):
        return np.ones_like(T)*intercept
    
    [T_min, T_max] = T_fit_bnds

    if cdd_bp < hdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp
        hdd_beta, cdd_beta = cdd_beta, hdd_beta
        hdd_k, cdd_k = cdd_k, hdd_k

    E_tot = np.empty_like(T)
    for n, Ti in enumerate(T):
        if (Ti < hdd_bp) or ((hdd_bp == cdd_bp) and (cdd_bp >= T_max)):   # Temperature is within the heating model
            T_bp = hdd_bp
            beta = -hdd_beta
            k = hdd_k

        elif (Ti > cdd_bp) or ((hdd_bp == cdd_bp) and (hdd_bp <= T_min)): # Temperature is within the cooling model
            T_bp = cdd_bp
            beta = cdd_beta
            k = -cdd_k
            
        else: # Temperature independent
            beta = 0

        # Evaluate
        if beta == 0: # tidd
            E_tot[n] = intercept

        elif k == 0: # c_hdd
            E_tot[n] = beta*(Ti - T_bp) + intercept

        else: # smoothed c_hdd
            c_hdd = beta*(Ti - T_bp) + intercept

            exp_interior = 1/k*(Ti - T_bp)
            exp_interior = np.clip(exp_interior, ln_min_pos_system_value, ln_max_pos_system_value)
            E_tot[n] = abs(beta*k)*(np.exp(exp_interior) - 1) + c_hdd

    return E_tot


@numba.jit(nopython=True, error_model='numpy', cache=numba_cache)
def get_full_model_x(model_key, x, T_min, T_max, T_min_seg, T_max_seg):
    if model_key == "hdd_tidd_cdd_smooth":
        [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept] = x

    elif model_key == "hdd_tidd_cdd":
        [hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept] = x
        hdd_k = cdd_k = 0

    elif model_key == "c_hdd_tidd_smooth":
        [c_hdd_bp, c_hdd_beta, c_hdd_k, intercept] = x
        hdd_bp = cdd_bp = c_hdd_bp

        if c_hdd_beta < 0:
            hdd_beta = -c_hdd_beta
            hdd_k = c_hdd_k
            cdd_beta = cdd_k = 0

        else:
            cdd_beta = c_hdd_beta
            cdd_k = c_hdd_k
            hdd_beta = hdd_k = 0

    elif model_key == "c_hdd_tidd":
        [c_hdd_bp, c_hdd_beta, intercept] = x

        if c_hdd_bp < T_min_seg:
            cdd_bp = hdd_bp = T_min
        elif c_hdd_bp > T_max_seg:
            cdd_bp = hdd_bp = T_max
        else:
            hdd_bp = cdd_bp = c_hdd_bp

        if c_hdd_beta < 0:
            hdd_beta = -c_hdd_beta
            cdd_beta = cdd_k = hdd_k = 0

        else:
            cdd_beta = c_hdd_beta
            hdd_beta = hdd_k = cdd_k = 0

    elif model_key == "tidd":
        [intercept] = x
        hdd_bp = hdd_beta = hdd_k = cdd_bp = cdd_beta = cdd_k = 0

    x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

    return fix_full_model_x(x, T_min, T_max)


@numba.jit(nopython=True, error_model='numpy', cache=numba_cache)
def fix_full_model_x(x, T_min_seg, T_max_seg):
    hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept = x

    # swap breakpoint order if they are reversed [hdd, cdd]
    if cdd_bp < hdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp
        hdd_beta, cdd_beta = cdd_beta, hdd_beta
        hdd_k, cdd_k = cdd_k, hdd_k

    # if there is a slope, but the breakpoint is at the end, it's a c_hdd_tidd model
    if hdd_bp != cdd_bp:
        if cdd_bp >= T_max_seg:
            cdd_beta = 0
        elif hdd_bp <= T_min_seg:
            hdd_beta = 0

    # if slopes are zero then smoothing is zero
    if hdd_beta == 0:
        hdd_k = 0

    if cdd_beta == 0:
        cdd_k = 0

    return [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]