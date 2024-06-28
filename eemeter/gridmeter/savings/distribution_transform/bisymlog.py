import numpy as np

from scipy.optimize import minimize_scalar
from scipy.stats import skew

from eemeter.common.adaptive_loss import IQR_outlier
from eemeter.gridmeter.savings.distribution_transform.misc import adaptive_weighted_mu_sigma



def _bisymlog_transform(x, C=1, rescale_quantile=None):
    idx = np.isfinite(x)   # only perform transformation on finite values
    y = np.empty_like(x)
    y[~idx] = np.nan

    if rescale_quantile is None:
        y[idx] = np.sign(x[idx])*np.log10(np.abs(x[idx]/C) + 1)

    else:
        # get prior quantiles for rescaling
        pq = np.quantile(x[idx], [rescale_quantile, 1 - rescale_quantile])
    
        y[idx] = np.sign(x[idx])*np.log10(np.abs(x[idx]/C) + 1)

        # get current quantiles for rescaling and set rescaling functions
        cq = np.quantile(y[idx], [rescale_quantile, 1 - rescale_quantile])

        # rescale to prior quantiles
        y[idx] = (y[idx] - cq[0])/np.diff(cq)*np.diff(pq) + pq[0]

    return y


def bisymlog_transform(x, rescale_quantile=None):
    def obj_fcn(X):
        C = 10**X

        xt = _bisymlog_transform(x, C=C, rescale_quantile=rescale_quantile)
        mu, sigma = adaptive_weighted_mu_sigma(xt, use_mean=False, rel_err=1E-4, abs_err=1E-4)
        xt = (xt - mu)/sigma

        xt_outliers = IQR_outlier(xt, sigma_threshold=3, quantile=0.05)
        xt = xt[(xt_outliers[0] < xt) & (xt < xt_outliers[1])]

        return np.abs(skew(xt))

    bnds = [-14, 6]

    res = minimize_scalar(obj_fcn, bounds=bnds, method='bounded')
    C = 10**res.x

    xt = _bisymlog_transform(x, C=C, rescale_quantile=rescale_quantile)
    mu, sigma = adaptive_weighted_mu_sigma(xt, use_mean=False, rel_err=1E-4, abs_err=1E-4)
    xt = (xt - mu)/sigma

    return xt