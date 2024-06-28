import numpy as np

from eemeter.gridmeter.savings.distribution_transform.misc import robust_mu_sigma


def robust_standardize(x, robust_type="huber_m_estimate"):
    mu, sigma = robust_mu_sigma(x, robust_type, c=1.5, tol=1e-08)
    x_std = (x - mu)/sigma

    return x_std