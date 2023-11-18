"""
functions for dealing with fitting to clusters
"""

from __future__ import annotations


import scipy.optimize
import scipy.spatial.distance

import numpy as np
import pandas as pd

from gridmeter._utils.adaptive_loss import adaptive_weights


def fit_to_clusters(t_ls, cp_ls, x0, agg_type: str):
    # agg_type = 'mean' # overwrite to force agg_type to be mean
    sigma = 2.698  # 1.5 IQR

    def obj_fcn_dec(t_ls, cp_ls, idx=None):
        if idx is not None:
            cp_ls = cp_ls[idx, :]

        def obj_fcn(x):
            resid = (t_ls - (cp_ls * x[:, None]).sum(axis=0)).flatten()

            weight, _, _ = adaptive_weights(
                x=resid, sigma=sigma, agg_type=agg_type  # type: ignore
            )

            wSSE = np.sum(weight * resid**2)

            return wSSE

        return obj_fcn

    def sum_to_one(x):
        zero = np.sum(x) - 1
        return zero

    x0 = np.array(x0).flatten()

    # only optimize if > 0.1%
    idx = np.argwhere(x0 >= 0.001).flatten()
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
    x[idx] = res.x

    return x


def t_meter_match(
    df_ls_t: pd.DataFrame, df_ls_clusters: pd.Series, agg_type: str, dist_metric: str
):
    if dist_metric == "manhattan":
        dist_metric = "cityblock"

    t_ls = df_ls_t.unstack().to_numpy()
    cp_ls = df_ls_clusters.unstack().to_numpy()

    # Get percent from each cluster
    # distances = cdist(t_ls_all, cp_ls, metric='euclidean')
    distances = scipy.spatial.distance.cdist(t_ls, cp_ls, metric=dist_metric)  # type: ignore
    distances_norm = (np.min(distances, axis=1) / distances.T).T
    distances_norm = (
        distances_norm**20
    )  # change this number to alter weights, larger centralizes the weight, smaller spreads them out
    distances_norm = (distances_norm.T / np.sum(distances_norm, axis=1)).T

    coeffs = fit_to_clusters(
        t_ls=t_ls, cp_ls=cp_ls, x0=distances_norm, agg_type=agg_type
    )

    t_ids = df_ls_t.index.get_level_values("id").unique()
    columns = [f"pct_cluster_{int(n)}" for n in np.arange(coeffs.shape[0])]
    df_t_coeffs = pd.DataFrame(coeffs[:, None].T, index=t_ids, columns=columns)

    return df_t_coeffs
