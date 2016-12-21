import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from scipy.stats import chi2

class CaltrackModel(object):
    ''' This class implements the two-stage modeling routine agreed upon
    as part of the Caltrack beta test. If fit_cdd is True, then all four
    candidate models (HDD+CDD, CDD-only, HDD-only, and Intercept-only) are 
    used in stage 1 estimation. If it's false, then only HDD-only and 
    Intercept-only are used. '''
    def __init__(self, fit_cdd=True):

        self.fit_cdd = fit_cdd
        self.model_freq = pd.tseries.frequencies.MonthEnd()
        self.params = None
        self.X = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None
        self.n = None
        self.input_data = None

    def __repr__(self):
        if self.fit_cdd: return ( 'Caltrack full' )
        else: return ( 'Caltrack HDD-only' )

    def fit(self, input_data):
        self.input_data = input_data
        df = input_data
        # Fit the intercept-only model
        int_formula = 'upd ~ 1'
        try:
            int_mod = smf.ols(formula=int_formula, data=df)
            int_res = int_mod.fit()
            int_rsquared = int_res.rsquared
            int_qualified = True
        except:
            int_rsquared, int_qualified = 0, False
    
        # CDD-only
        cdd_formula = 'upd ~ CDD'
        try:
            if not self.fit_cdd: assert False
            cdd_mod = smf.ols(formula=cdd_formula, data=df)
            cdd_res = cdd_mod.fit()
            cdd_rsquared = cdd_res.rsquared
            cdd_qualified = (cdd_res.params['Intercept'] >= 0) and (cdd_res.params['CDD'] >= 0) and \
                            (cdd_res.pvalues['Intercept'] < 0.1) and (cdd_res.pvalues['CDD'] < 0.1)
        except:
            cdd_rsquared, cdd_qualified = 0, False
    
        # HDD-only
        hdd_formula = 'upd ~ HDD'
        try:
            hdd_mod = smf.ols(formula=hdd_formula, data=df)
            hdd_res = hdd_mod.fit()
            hdd_rsquared = hdd_res.rsquared
            hdd_qualified = (hdd_res.params['Intercept'] >= 0) and (hdd_res.params['HDD'] >= 0) and \
                            (hdd_res.pvalues['Intercept'] < 0.1) and (hdd_res.pvalues['HDD'] < 0.1)
        except:
            hdd_rsquared, hdd_qualified = 0, False

        # CDD+HDD
        full_formula = 'upd ~ CDD + HDD'
        try:
            if not self.fit_cdd: assert False
            full_mod = smf.ols(formula=full_formula, data=df)
            full_res = full_mod.fit()
            full_rsquared = full_res.rsquared
            full_qualified = (full_res.params['Intercept'] >= 0) and (full_res.params['HDD'] >= 0) and (full_res.params['CDD'] >= 0) and \
                             (full_res.pvalues['Intercept'] < 0.1) and (full_res.pvalues['CDD'] < 0.1) and (full_res.pvalues['HDD'] < 0.1)
        except:
            full_rsquared, full_qualified = 0, False
    
        # Now we take the best qualified model.
        if (full_qualified or hdd_qualified or cdd_qualified or int_qualified) == False: 
            raise ValueError("No candidate model fit to data successfully")
            return None
        if full_qualified and full_rsquared > \
            max([int(hdd_qualified)*hdd_rsquared,int(cdd_qualified)*cdd_rsquared,int(int_qualified)*int_rsquared]):
            # Use the full model
            self.y,self.X = patsy.dmatrices(full_formula, df, return_type='dataframe')
            self.estimated = full_res.fittedvalues
            self.r2, self.rmse = full_rsquared, np.sqrt(full_res.mse_total)
            self.model_obj, self.model_res, formula = full_mod, full_res, full_formula
        elif hdd_qualified and hdd_rsquared > \
            max([int(full_qualified)*full_rsquared,int(cdd_qualified)*cdd_rsquared,int(int_qualified)*int_rsquared]):
            # Use HDD-only
            self.y,self.X = patsy.dmatrices(hdd_formula, df, return_type='dataframe')
            self.estimated = hdd_res.fittedvalues
            self.r2, self.rmse = hdd_rsquared, np.sqrt(hdd_res.mse_total)
            self.model_obj, self.model_res, formula = hdd_mod, hdd_res, hdd_formula
        elif cdd_qualified and cdd_rsquared > \
            max([int(full_qualified)*full_rsquared,int(hdd_qualified)*hdd_rsquared,int(int_qualified)*int_rsquared]):
            # Use CDD-only
            self.y,self.X = patsy.dmatrices(cdd_formula, df, return_type='dataframe')
            self.estimated = cdd_res.fittedvalues
            self.r2, self.rmse = cdd_rsquared, np.sqrt(cdd_res.mse_total)
            self.model_obj, self.model_res, formula = cdd_mod, cdd_res, cdd_formula
        else:
            # Use Intercept-only
            self.y,self.X = patsy.dmatrices(int_formula, df, return_type='dataframe')
            self.estimated = int_res.fittedvalues
            self.r2, self.rmse = int_rsquared, np.sqrt(int_res.mse_total)
            self.model_obj, self.model_res, formula = int_mod, int_res, int_formula

        if self.y.mean != 0:
            self.cvrmse = self.rmse / float(self.y.values.ravel().mean())
        else:
            self.cvrmse = np.nan

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

        #self.plot()

        self.params = {
            "coefficients": self.model_res.params, 
            "X_design_info": self.X.design_info,
            "formula": formula,
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

    def predict(self, demand_fixture_data, params=None, summed=True):
        ''' Predicts across index using fitted model params

        Parameters
        ----------
        demand_fixture_data : pandas.DataFrame
            Formatted input data as returned by
            :code:`CaltrackFormatter.create_demand_fixture()`
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

        design_info = params["X_design_info"]

        (X,) = patsy.build_design_matrices([design_info],
                                           demand_fixture_data,
                                           return_type='dataframe')

        predicted = self.model_res.predict(X)
        lower = np.zeros(predicted.shape)
        upper = np.zeros(predicted.shape)
        # Get parameter covariance matrix
        cov = self.model_res.cov_params()
        # Get prediction errors for each data point
        prediction_var = self.model_res.mse_resid * (X * np.dot(cov,X.T).T).sum(1)
        predicted_baseline_use, predicted_baseline_use_var = 0.0, 0.0

        # Sum them up using the number of days in the demand fixture.
        for i in range(len(demand_fixture_data.index)):
            predicted[i] = predicted[i] * demand_fixture_data.ndays[i]
            predicted_baseline_use = predicted_baseline_use + predicted[i]
            thisvar = prediction_var[i] * demand_fixture_data.ndays[i]
            predicted_baseline_use_var = predicted_baseline_use_var + thisvar
            lower[i] = np.sqrt(thisvar) * 1.959964 / 2
            upper[i] = np.sqrt(thisvar) * 1.959964 / 2

        if summed:
            return predicted_baseline_use, np.sqrt(predicted_baseline_use_var)*1.959964 / 2, \
                   np.sqrt(predicted_baseline_use_var)*1.959964 / 2
        else:
            predicted = pd.Series(predicted, index=X.index)
            lower = pd.Series(lower, index=X.index)
            upper = pd.Series(upper, index=X.index)
            return predicted, upper, lower

    def calc_gross(self):
        gross = 0.0
        for i in range(len(self.input_data.index)):
            if np.isfinite(self.input_data.upd):
                gross = gross + self.input_data.upd[i] * self.input_data.ndays[i]
        return gross

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
