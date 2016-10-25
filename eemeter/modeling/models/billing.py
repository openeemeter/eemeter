import warnings


import numpy as np
import pandas as pd
import patsy
from scipy.stats import chi2
from sklearn import linear_model


class BillingElasticNetCVModel():
    ''' Linear regression of energy values against CDD/HDD with elastic net
    regularization.

    Parameters
    ----------
    cooling_base_temp : float
        Base temperature (degrees F) used in calculating cooling degree days.
    heating_base_temp : float
        Base temperature (degrees F) used in calculating heating degree days.
    '''

    def __init__(self, cooling_base_temp, heating_base_temp):

        self.cooling_base_temp = cooling_base_temp
        self.heating_base_temp = heating_base_temp

        self.formula = 'energy ~ 1 + CDD + HDD + CDD:HDD'

        self.l1_ratio = 0.5
        self.params = None
        self.upper = None
        self.lower = None
        self.X = None
        self.y = None
        self.r2 = None
        self.estimated = None
        self.rmse = None
        self.cvrmse = None
        self.n = None

    def _cdd(self, temperature_data):
        if 'hourly' in temperature_data.index.names:
            cdd = np.maximum((temperature_data - self.cooling_base_temp)
                             .groupby(level='period').sum()[0] / 24.0, 0.0)
        elif 'daily' in temperature_data.index.names:
            cdd = np.maximum((temperature_data - self.cooling_base_temp)
                             .groupby(level='period').sum()[0], 0.0)
        else:
            cdd = []
        return cdd

    def _hdd(self, temperature_data):
        if 'hourly' in temperature_data.index.names:
            hdd = np.maximum((self.heating_base_temp - temperature_data)
                             .groupby(level='period').sum()[0] / 24.0, 0.0)
        elif 'daily' in temperature_data.index.names:
            hdd = np.maximum((self.heating_base_temp - temperature_data)
                             .groupby(level='period').sum()[0], 0.0)
        else:
            hdd = []
        return hdd

    def fit(self, input_data):
        ''' Fits a model to the input data.

        Parameters
        ----------
        input_data : pandas.DataFrame
            Formatted input data as returned by
            :code:`ModelDataBillingFormatter.create_input()`

        Returns
        -------
        out : dict
            Results of this model fit:

            - :code:`"r2"`: R-squared value from this fit.
            - :code:`"model_params"`: Fitted parameters.

              - :code:`X_design_matrix`: patsy design matrix used in
                formatting design matrix.
              - :code:`formula`: patsy formula used in creating design matrix.
              - :code:`coefficients`: ElasticNetCV coefficients.
              - :code:`intercept`: ElasticNetCV intercept.

            - :code:`"rmse"`: Root mean square error
            - :code:`"cvrmse"`: Normalized root mean square error
              (Coefficient of variation of root mean square error).
            - :code:`"upper"`: self.upper,
            - :code:`"lower"`: self.lower,
            - :code:`"n"`: self.n
        '''
        trace_data, temperature_data = input_data

        cdd = self._cdd(temperature_data)
        hdd = self._hdd(temperature_data)
        model_data = pd.DataFrame(
            {'energy': trace_data.iloc[:-1], 'CDD': cdd, 'HDD': hdd},
            columns=['energy', 'CDD', 'HDD'])

        model_data = model_data.dropna()

        y, X = patsy.dmatrices(self.formula, model_data,
                               return_type='dataframe')

        model_obj = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,
                                              fit_intercept=False)
        model_obj.fit(X, y.values.ravel())

        estimated = pd.Series(model_obj.predict(X),
                              index=model_data.energy.index)

        self.X = X
        self.y = y
        self.estimated = estimated

        r2 = model_obj.score(X, y)
        rmse = ((y.values.ravel() - estimated)**2).mean()**.5

        if y.mean != 0:
            cvrmse = rmse / float(y.values.ravel().mean())
        else:
            cvrmse = np.nan

        self.r2 = r2
        self.rmse = rmse
        self.cvrmse = cvrmse

        # For justification of these 95% confidence intervals, based on rmse,
        # see http://stats.stackexchange.com/questions/78079/
        #     confidence-interval-of-rmse
        #
        # > Let xi be your true value for the ith data point and xhat_i the
        # >   estimated value.
        # > If we assume that the differences between the estimated and
        # > true values have
        # >
        # > 1. mean zero (i.e. the xhat_i are distributed around xi)
        # > 2. follow a Normal distribution
        # > 3. and all have the same standard deviation sigma
        # > then you really want a confidence interval for sigma
        # > ...
        #
        # We might decide these assumptions don't hold.

        n = self.estimated.shape[0]

        c1, c2 = chi2.ppf([0.025, 1-0.025], n)
        self.lower = np.sqrt(n/c2) * self.rmse
        self.upper = np.sqrt(n/c1) * self.rmse
        self.n = n

        self.plot()

        self.params = {
            "coefficients": model_obj.coef_,
            "intercept": model_obj.intercept_,
            "X_design_info": X.design_info,
            "formula": self.formula,
        }

        output = {
            "r2": self.r2,
            "model_params": self.params,
            "rmse": self.rmse,
            "cvrmse": self.cvrmse,
            "upper": self.upper,
            "lower": self.lower,
            "n": self.n
        }
        return output

    def predict(self, demand_fixture_data, params=None):
        ''' Predicts across index using fitted model params

        Parameters
        ----------
        demand_fixture_data : pandas.DataFrame
            Formatted input data as returned by
            :code:`ModelDataBillingFormatter.create_demand_fixture()`
        params : dict, default None
            Parameters found during model fit. If None, `.fit()` must be called
            before this method can be used.

              - :code:`X_design_matrix`: patsy design matrix used in
                formatting design matrix.
              - :code:`formula`: patsy formula used in creating design matrix.
              - :code:`coefficients`: ElasticNetCV coefficients.
              - :code:`intercept`: ElasticNetCV intercept.

        Returns
        -------
        output : pandas.DataFrame
            Dataframe of energy values as given by the fitted model across the
            index given in :code:`demand_fixture_data`.
        '''
        # needs only tempF
        if params is None:
            params = self.params

        model_data = demand_fixture_data.resample(
            pd.tseries.frequencies.Day()).agg({'tempF': np.mean})

        model_data.loc[:, 'CDD'] = np.maximum(model_data.tempF -
                                              self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.maximum(self.heating_base_temp -
                                              model_data.tempF, 0.)

        design_info = params["X_design_info"]

        (X,) = patsy.build_design_matrices([design_info],
                                           model_data,
                                           return_type='dataframe')

        model_obj = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,
                                              fit_intercept=False)

        model_obj.coef_ = params["coefficients"]
        model_obj.intercept_ = params["intercept"]

        predicted = pd.Series(model_obj.predict(X), index=X.index)

        # add NaNs back in
        predicted = predicted.reindex(model_data.index)

        return predicted, self.lower, self.upper

    def plot(self):
        ''' Plots fit against input data. Should not be run before the
        :code:`.fit(` method.
        '''

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Cannot plot - no matplotlib.")
            return None

        plt.title("actual v. estimated w/ 95% confidence")

        self.estimated.plot(color='b', alpha=0.7)

        plt.fill_between(self.estimated.index.to_datetime(),
                         self.estimated + self.upper,
                         self.estimated - self.lower,
                         color='b', alpha=0.3)

        pd.Series(self.y.values.ravel(), index=self.estimated.index).plot(
                  color='k', linewidth=1.5)

        plt.show()
