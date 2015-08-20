from .base import MeterBase

import numpy as np

class CVRMSE(MeterBase):
    """Coefficient of Variation of Root-Mean-Square Error for a model fit.
    """
    def evaluate_raw(self, y, y_hat, params, **kwargs):
        """Evaluates the Coefficient of Variation of Root-Mean-Square Error of
        a model fit.

        Parameters
        ----------
        y : array_like
            Observed values.
        y_hat : array_like
            Estimated values.
        params : eemeter.models.parameters.ParameterType
            Model parameters (used only for counting the number of parameters).

        Returns
        -------
        out : dict
            - "cvrmse" : the calculated CVRMSE metric.
        """
        y_bar = np.nanmean(y)
        n = len(y)
        p = len(params.to_list())
        cvrmse = 100 * (np.nansum((y - y_hat)**2) / (n - p) )**.5 / y_bar
        return {"cvrmse": cvrmse}

class RMSE(MeterBase):
    """Compute the root-mean-square error (sometimes referred to as
    root-mean-square deviation, or RMSD) of observed samples and estimated
    values.
    """
    def evaluate_raw(self, y, y_hat, **kwargs):
        """Evaluates the Coefficient of Variation of Root-Mean-Square Error of
        a model fit.

        Parameters
        ----------
        y : array_like
            Observed values.
        y_hat : array_like
            Estimated values.

        Returns
        -------
        out : dict
            - "rmse" : the calculated RMSE metric.
        """
        n = len(y)
        rmse = (np.nansum((y - y_hat)**2) / n )**.5
        return {"rmse": rmse}

class RSquared(MeterBase):
    """Compute the :math:`r^2` metric (coefficient of determination) of observed
    samples and estimated values. Used to measure the fitness of a model.
    """
    def evaluate_raw(self, y, y_hat, **kwargs):
        """Evaluates the :math:`r^2` fitness metric for particular samples

        Parameters
        ----------
        y : array_like
            Observed values.
        y_hat : array_like
            Estimated values.

        Returns
        -------
        out : dict
            - "r_squared" : the calculated :math:`r^2` fitness metric.
        """
        y_bar = np.nanmean(y)
        ss_residual = np.nansum( (y - y_hat)**2 )
        ss_total = np.nansum( (y - y_bar)**2 )
        r_squared = 1 - ss_residual / ss_total

        return {"r_squared": r_squared}
