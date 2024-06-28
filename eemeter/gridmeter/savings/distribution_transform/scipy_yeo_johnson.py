from scipy.stats import yeojohnson

from eemeter.gridmeter.savings.distribution_transform.misc import robust_mu_sigma
from eemeter.gridmeter.savings.distribution_transform.standardize import robust_standardize


def scipy_YJ_transform(x, robust_type="huber_m_estimate"):
    x_std, _ = yeojohnson(x, lmbda=None)
    x_std = robust_standardize(x_std, robust_type)

    return x_std