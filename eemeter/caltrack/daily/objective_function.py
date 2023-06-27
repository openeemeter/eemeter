from copy import deepcopy as copy

import numpy as np

from eemeter.caltrack.daily.utilities.adaptive_loss import remove_outliers
from eemeter.caltrack.daily.utilities.adaptive_loss import rolling_C
from eemeter.caltrack.daily.utilities.adaptive_loss import adaptive_weights

from eemeter.caltrack.daily.utilities.utils import OoM
from eemeter.caltrack.daily.utilities.utils import fast_std as stdev

# from eemeter.caltrack.daily.utilities.utils_derivative import numerical_jacobian


def get_idx(A, B):
    idx = []
    for item in A:
        try:
            idx.extend([B.index(txt) for txt in B if item in txt])
        except:
            continue

    idx.sort()

    return idx


def no_weights_obj_fcn(X, aux_inputs):
    model_fcn, obs, idx_bp = aux_inputs

    # flip breakpoints if they are not in the correct order
    if (len(idx_bp) > 1) and (X[idx_bp[0]] > X[idx_bp[1]]):
        X[idx_bp[1]], X[idx_bp[0]] = X[idx_bp[0]], X[idx_bp[1]]

    model = model_fcn(X)
    resid = model - obs

    SSE = np.sum(resid**2)

    return SSE


def model_fcn_dec(model_fcn_full, T_fit_bnds, T):
    def model_fcn_X_only(X):
        return model_fcn_full(*X, T_fit_bnds, T)

    return model_fcn_X_only


def obj_fcn_decorator(
    model_fcn_full,
    weight_fcn,
    TSS_fcn,
    T,
    obs,
    settings,
    alpha=2.0,
    C=None,
    coef_id=[],
    initial_fit=True,
):
    N = np.shape(obs)[0]
    N_min = settings.segment_minimum_count  # N minimum for a sloped segment
    sigma = 2.698  # 1.5 IQR
    quantile = 0.25
    min_weight = 0.00
    # window = 0.25
    # step = 0.0
    window = 0.15
    step = 1.0
    # window = 0.1
    # step = 0.4

    T_fit_bnds = np.array([np.min(T), np.max(T)])

    model_fcn = model_fcn_dec(model_fcn_full, T_fit_bnds, T)

    lasso_a = settings.regularization_percent_lasso * settings.regularization_alpha
    ridge_a = (
        1 - settings.regularization_percent_lasso
    ) * settings.regularization_alpha

    idx_k = get_idx(["dd_k"], coef_id)
    idx_beta = get_idx(["dd_beta"], coef_id)
    idx_bp = get_idx(["dd_bp"], coef_id)
    # idx_reg = get_idx(["dd_beta", "dd_k"], coef_id) # drop bps and intercept from regularization
    idx_reg = get_idx(
        ["dd_beta", "dd_k", "dd_bp"], coef_id
    )  # drop intercept from regularization

    # sometimes rolling C fails, when it fails, fall back to weight function version
    if C is None:
        C_list = [C]
    else:
        C_list = [C, None]

    def elastic_net_penalty(X, T_sorted, obs_sorted, weight_sorted):
        # Elastic net
        X_enet = np.array(X).copy()

        ## Scale break points ##
        if len(idx_bp) > 0:
            T_bnds = np.array([T_sorted[0], T_sorted[-1]])

            X_enet[idx_bp] = [np.min(np.abs(X_enet[idx] - T_bnds)) for idx in idx_bp]

            if len(idx_bp) == 2:
                X_enet[idx_bp] += (T_bnds[1] - T_bnds[0]) / 2
                # X_enet[idx_bp] /= 2

        # Find idx for regions
        if len(idx_bp) == 2:
            [hdd_bp, cdd_bp] = X[idx_bp]

            idx_hdd = np.argwhere(T_sorted < hdd_bp).flatten()
            idx_tidd = np.argwhere(
                (hdd_bp <= T_sorted) & (T_sorted <= cdd_bp)
            ).flatten()
            idx_cdd = np.argwhere(cdd_bp < T_sorted).flatten()

        elif len(idx_bp) == 1:
            bp = X[idx_bp]
            if X_enet[idx_beta] < 0:  # HDD_TIDD
                idx_hdd = np.argwhere(T_sorted <= bp).flatten()
                idx_tidd = np.argwhere(bp < T_sorted).flatten()
                idx_cdd = np.array([])

            else:
                idx_hdd = np.array([])  # CDD_TIDD
                idx_tidd = np.argwhere(T_sorted < bp).flatten()
                idx_cdd = np.argwhere(bp <= T_sorted).flatten()

        else:
            idx_hdd = np.array([])
            idx_tidd = np.arange(0, len(T_sorted))
            idx_cdd = np.array([])

        len_hdd = len(idx_hdd)
        len_tidd = len(idx_tidd)
        len_cdd = len(idx_cdd)

        # combine tidd with hdd/cdd if cdd/hdd are large enough to get stdev
        if (len_hdd < N_min) and (len_cdd >= N_min):
            idx_hdd = np.hstack([idx_hdd, idx_tidd])
        elif (len_hdd >= N_min) and (len_cdd < N_min):
            idx_cdd = np.hstack([idx_tidd, idx_cdd])

        # change to idx_hdd and idx_cdd to int arrays
        idx_hdd = idx_hdd.astype(int)
        idx_cdd = idx_cdd.astype(int)

        ## Normalize slopes ##
        # calculate stdevs
        if (len(idx_bp) == 2) and (len(idx_hdd) >= N_min) and (len(idx_cdd) >= N_min):
            N_beta = np.array([len_hdd, len_cdd])
            T_stdev = np.array(
                [
                    stdev(T_sorted[idx_hdd], weights=weight_sorted[idx_hdd]),
                    stdev(T_sorted[idx_cdd], weights=weight_sorted[idx_cdd]),
                ]
            )
            obs_stdev = np.array(
                [
                    stdev(obs_sorted[idx_hdd], weights=weight_sorted[idx_hdd]),
                    stdev(obs_sorted[idx_cdd], weights=weight_sorted[idx_cdd]),
                ]
            )

        elif (len(idx_bp) == 1) and (len(idx_hdd) >= N_min):
            N_beta = np.array([len_hdd])
            T_stdev = stdev(T_sorted[idx_hdd], weights=weight_sorted[idx_hdd])
            obs_stdev = stdev(obs_sorted[idx_hdd], weights=weight_sorted[idx_hdd])

        elif (len(idx_bp) == 1) and (len(idx_cdd) >= N_min):
            N_beta = np.array([len_cdd])
            T_stdev = stdev(T_sorted[idx_cdd], weights=weight_sorted[idx_cdd])
            obs_stdev = stdev(obs_sorted[idx_cdd], weights=weight_sorted[idx_cdd])

        else:
            N_beta = np.array([len_tidd])
            T_stdev = stdev(T_sorted, weights=weight_sorted)
            obs_stdev = stdev(obs_sorted, weights=weight_sorted)

        X_enet[idx_beta] *= T_stdev / obs_stdev

        # add penalty to slope for not having enough datapoints
        X_enet[idx_beta] = np.where(
            N_beta < N_min, X_enet[idx_beta] * 1e30, X_enet[idx_beta]
        )

        ## Scale smoothing parameter ##
        if len(idx_k) > 0:  # reducing X_enet size allows for more smoothing
            X_enet[idx_k] = X[idx_k]

            if (len(idx_k) == 2) and (np.sum(X_enet[idx_k]) > 1):
                X_enet[idx_k] /= np.sum(X_enet[idx_k])

            X_enet[idx_k] *= (
                X_enet[idx_beta] / 2
            )  # uncertain what to divide by, this seems to work well

        X_enet = X_enet[idx_reg]

        if ridge_a == 0:
            penalty = lasso_a * np.linalg.norm(X_enet, 1)
        else:
            penalty = lasso_a * np.linalg.norm(X_enet, 1) + ridge_a * np.linalg.norm(
                X_enet, 2
            )

        return penalty

    def obj_fcn(X, grad=[], optimize_flag=True):
        X = np.array(X)

        model = model_fcn(X)
        idx_sorted = np.argsort(T).flatten()
        idx_initial = np.argsort(idx_sorted).flatten()
        resid = model - obs

        T_sorted = T[idx_sorted]
        # model_sorted = model[idx_sorted]
        obs_sorted = obs[idx_sorted]
        resid_sorted = resid[idx_sorted]

        for C in C_list:
            if C is None:
                weight_sorted, c, a = weight_fcn(
                    *X, T_sorted, resid_sorted, sigma, quantile, alpha, min_weight
                )

            else:
                try:  # I know it's ugly, but it won't break?
                    resid_no_outlier, _ = remove_outliers(
                        resid, sigma_threshold=sigma, quantile=0.25
                    )

                    # mu = np.median(np.abs(resid_no_outlier))
                    mu = np.median(resid_no_outlier)

                    if isinstance(C, float):
                        c = C
                    else:
                        c = rolling_C(
                            T_sorted, resid_sorted, mu, sigma, 0.25, window, step
                        )

                    weight_sorted, a = adaptive_weights(
                        resid_sorted, mu=mu, c=c, alpha=alpha, min_weight=min_weight
                    )

                except:
                    pass

        weight = weight_sorted[idx_initial]
        wSSE = np.sum(weight * resid**2)
        loss = wSSE / N

        if settings.regularization_alpha != 0:
            loss += elastic_net_penalty(X, T_sorted, obs_sorted, weight_sorted)

        if optimize_flag:
            return loss

        else:
            if ("r_squared" in settings.split_selection_criteria) and callable(TSS_fcn):
                TSS = TSS_fcn(*X, T_sorted, obs_sorted)
            else:
                TSS = wSSE

            if initial_fit:
                jac = None
            else:
                eps = 10 ** (OoM(X, method="floor") - 2)
                X_lower = X - eps
                X_upper = X + eps

                # select correct finite difference scheme based on variable type and value
                # NOTE: finite differencing was not returning great results. Looking into switching to JAX autodiff
                fd_type = ["central"] * len(X)
                for i in range(len(X)):
                    if i in idx_k:
                        if X_lower[i] < 0:
                            fd_type[i] = "forward"
                        elif X_upper[i] > 1:
                            fd_type[i] = "backward"
                    elif i in idx_beta:
                        if (X[i] > 0) and (X_lower[i] < 0):
                            fd_type[i] = "forward"
                        elif (X[i] < 0) and (X_upper[i] > 0):
                            fd_type[i] = "backward"
                    elif i in idx_bp:
                        if X_lower[i] < T_sorted[0]:
                            fd_type[i] = "forward"
                        elif X_lower[i] > T_sorted[-1]:
                            fd_type[i] = "backward"

                # https://stackoverflow.com/questions/70572362/compute-efficiently-hessian-matrices-in-jax
                # hess = jit(jacfwd(jacrev(no_weights_obj_fcn), has_aux=True), has_aux=True)(X, [model_fcn, obs, idx_bp])
                # print(hess)
                # obj_grad_fcn = lambda X: no_weights_obj_fcn(X, [model_fcn, obs, idx_bp])

                # jac = numerical_jacobian(obj_grad_fcn, X, dx=eps, fd_type=fd_type)
                jac = None

            return X, loss, TSS, T, model, weight, resid, jac, np.mean(a), c

    return obj_fcn
