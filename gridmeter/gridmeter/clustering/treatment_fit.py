"""
functions for dealing with fitting to clusters
"""

from __future__ import annotations


import scipy.optimize
import scipy.spatial.distance

import numpy as np
import pandas as pd

from gridmeter._utils import (
    adaptive_loss as _adaptive_loss,
    multiprocessing as _multiprocessing,
)
from gridmeter.clustering import settings as _settings


def fit_to_clusters(
    t_ls, 
    cp_ls, 
    x0, 
    s: _settings.Settings,
):
    # _AGG_TYPE = 'mean' # overwrite to force agg_type to be mean
    _AGG_TYPE = s.AGG_TYPE
    _ALPHA = s._TREATMENT_MATCH_LOSS_ALPHA
    _SIGMA = 2.698  # 1.5 IQR
    _MIN_PCT_CLUSTER = 1E-6
    
    def _remove_small_x(x: np.ndarray):
        # remove small values and normalize to 1
        x[x < _MIN_PCT_CLUSTER] = 0
        x /= np.sum(x)

        return x

    def obj_fcn_dec(t_ls, cp_ls, idx=None):
        if idx is not None:
            cp_ls = cp_ls[idx, :]

        def obj_fcn(x):
            x = _remove_small_x(x)
            resid = (t_ls - (cp_ls * x[:, None]).sum(axis=0)).flatten()

            if _ALPHA == 2:
                wSSE = np.sum(resid**2)

            else:
                weight, _, _ = _adaptive_loss.adaptive_weights(
                    x=resid,
                    alpha=_ALPHA,
                    sigma=_SIGMA, 
                    agg_type=_AGG_TYPE,

                ) # type: ignore

                wSSE = np.sum(weight * resid**2)

            return wSSE

        return obj_fcn

    def sum_to_one(x):
        zero = np.sum(x) - 1
        return zero

    x0 = np.array(x0).flatten()

    # only optimize if >= _MIN_PCT_CLUSTER
    idx = np.argwhere(x0 >= _MIN_PCT_CLUSTER).flatten()
    if len(idx) == 0:
        idx = np.arange(0, len(x0))

    x0_n = x0[idx]

    bnds = np.repeat(np.array([0, 1])[:, None], x0_n.shape[0], axis=1).T
    const = [{"type": "eq", "fun": sum_to_one}]

    res = scipy.optimize.minimize(
        obj_fcn_dec(t_ls, cp_ls, idx),
        x0_n,
        bounds=bnds,
        constraints=const,
        method="SLSQP",
    )  # trust-constr, SLSQP
    # res = minimize(obj_fcn, x0, bounds=bnds, method='SLSQP') # trust-constr, SLSQP, L-BFGS-B
    # res = differential_evolution(obj_fcn, bnds, maxiter=100)
    # res = basinhopping(obj_fcn, x0, niter=10, minimizer_kwargs={'bounds': bnds, 'method': 'Powell'})

    x = np.zeros_like(x0)
    x[idx] = _remove_small_x(res.x)

    return x

def fit_to_clusters_dec(args_list):
    return fit_to_clusters(*args_list)


# TODO: this is not a fast way to do this, could be parallelized
def _match_treatment_to_cluster(
    df_ls_t: pd.DataFrame, 
    df_ls_cluster: pd.Series, 
    s: _settings.Settings
):

    t_ls = df_ls_t.to_numpy()
    cp_ls = df_ls_cluster.to_numpy()

    # Get percent from each cluster
    distances = scipy.spatial.distance.cdist(t_ls, cp_ls, metric="euclidean")  # type: ignore
    distances_norm = (np.min(distances, axis=1) / distances.T).T
    # change this number (20) to alter weights, larger centralizes the weight, smaller spreads them out
    distances_norm = (distances_norm**20)  
    distances_norm = (distances_norm.T / np.sum(distances_norm, axis=1)).T

    args_list = []
    for n, t_id in enumerate(df_ls_t.index):
        t_id_ls = t_ls[n, :]
        x0 = distances_norm[n, :]

        args_list.append([t_id_ls, cp_ls, x0, s])

    coeffs = _multiprocessing._execute_with_mp(
        fit_to_clusters_dec, 
        args_list, 
        use_mp=s.USE_MULTIPROCESSING
    )

    coeffs = np.vstack(coeffs)

    # Create dataframe
    t_ids = df_ls_t.index
    columns = [f"pct_cluster_{int(n)}" for n in df_ls_cluster.index]
    df_t_coeffs = pd.DataFrame(coeffs, index=t_ids, columns=columns)

    return df_t_coeffs