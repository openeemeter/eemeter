from copy import deepcopy as copy

import numpy as np

from statsmodels.robust.scale import Huber as huber_m_estimate

from eemeter.common.utils import MAD_k
from eemeter.common.adaptive_loss import adaptive_weights, weighted_quantile


def robust_mu_sigma(x, robust_type="huber_m_estimate", **kwargs):
    if robust_type == "huber_m_estimate":
        mu, sigma = huber_m_estimate(maxiter=50, **kwargs)(x)

    elif robust_type == "adaptive_weighted":
        mu, sigma = adaptive_weighted_mu_sigma(x, **kwargs)

    return mu, sigma


def adaptive_weighted_mu_sigma(x, use_mean=False, rel_err=1E-4, abs_err=1E-4):
    mu = np.median(x)
    sigma = np.median(np.abs(x - mu))*MAD_k

    for n in range(10):
        mu_prior = copy(mu)
        sigma_prior = copy(sigma)
        weight = adaptive_weights(x, mu=mu_prior, sigma=sigma_prior)[0]
        if use_mean:
            mu = np.sum(weight*x)/np.sum(weight)
            sigma = np.sum(weight*(x - mu)**2)/np.sum(weight)

        else:
            mu = weighted_quantile(x, 0.5, weights=weight)
            sigma = weighted_quantile(np.abs(x - mu), 0.5, weights=weight)*MAD_k

        max_abs_err = np.max(np.abs([(mu - mu_prior), (sigma - sigma_prior)]))
        max_rel_err = np.max(np.abs([(mu - mu_prior)/mu_prior, (sigma - sigma_prior)/sigma_prior]))

        if (max_rel_err < rel_err) | (max_abs_err < abs_err):
            break

    if sigma == 0:
        sigma = 1

    return mu, sigma