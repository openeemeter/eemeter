"""
functions for dealing with fitting to clusters
"""

from __future__ import annotations


import scipy.optimize
import scipy.spatial.distance

import numpy as np
import pandas as pd

from gridmeter._utils.adaptive_loss import adaptive_weights
from gridmeter._clustering import (
    transform as _transform,
)


def _transform_treatment_loadshape(df: pd.DataFrame):
    """
    transforms a dataframe meant to be treatment loadshapes

    It can work either on a dataframe containing all treatment loadshapes
    or a single loadshape.

    Meant to be used on treatment matching as the transform is needed to occur as part of the matching process
    """
    # df_list: list[pd.DataFrame] = []
    # # TODO: This is slow. Need to vectorize or use apply?
    # for _id, data in df.iterrows():
    #     transformed_data = _transform._normalize_loadshape(
    #         ls_arr=data.values
    #     )
    #     df_list.append(transformed_data)
    # 
    # df_transformed = pd.concat(df_list).to_frame(name="ls")  # type: ignore

    df_transformed = df.apply(_transform._normalize_loadshape, axis=1)

    return df_transformed


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


# TODO: this is not a fast way to do this, could be parallelized
def _match_treatment_to_cluster(
    df_ls_t: pd.DataFrame, 
    df_ls_cluster: pd.Series, 
    agg_type: str, 
    dist_metric: str
):   
    if dist_metric == "manhattan":
        dist_metric = "cityblock"

    t_ls = df_ls_t.to_numpy()
    cp_ls = df_ls_cluster.to_numpy()

    # Get percent from each cluster
    distances = scipy.spatial.distance.cdist(t_ls, cp_ls, metric=dist_metric)  # type: ignore
    distances_norm = (np.min(distances, axis=1) / distances.T).T
    # change this number (20) to alter weights, larger centralizes the weight, smaller spreads them out
    distances_norm = (distances_norm**20)  
    distances_norm = (distances_norm.T / np.sum(distances_norm, axis=1)).T

    coeffs = []
    for n, t_id in enumerate(df_ls_t.index):
        t_id_ls = t_ls[n, :]
        x0 = distances_norm[n, :]

        id_coeffs = fit_to_clusters(t_ls=t_id_ls, cp_ls=cp_ls, x0=x0, agg_type=agg_type)

        coeffs.append(id_coeffs)

    coeffs = np.vstack(coeffs)

    # Create dataframe
    t_ids = df_ls_t.index
    columns = [f"pct_cluster_{int(n)}" for n in df_ls_cluster.index]
    df_t_coeffs = pd.DataFrame(coeffs, index=t_ids, columns=columns)

    return df_t_coeffs