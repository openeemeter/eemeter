import warnings

import calendar
import numpy as np
import pandas as pd
import patsy
from scipy.stats import chi2
import scipy.optimize
import statsmodels.formula.api as smf

class BillingNNLSModel():
    def __init__(self, cooling_base_temp, heating_base_temp, heating_only=False):

        self.cooling_base_temp = cooling_base_temp
        self.heating_base_temp = heating_base_temp
        self.heating_only = heating_only

        self.formula = 'energy ~ CDD + HDD'

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

    def _make_monthly(self, df):
        ''' This can and should be vectorized, but we're doing it pedantically
            now so as to not screw it up. '''
	df = copy.deepcopy(df_in)
	output_index = df.resample('M').apply(sum).index
        ndays, usage, upd, cdd, hdd = [0], [0], [0], [0], [0]
        this_yr, this_mo = output_index[0].year, output_index[0].month
        for idx, row in df.iterrows():
           if this_yr!=idx.year or this_mo!=idx.month:
               ndays.append(0)
               usage.append(0)
               upd.append(0)
               cdd.append(0)
               hdd.append(0)
               this_yr, this_mo = idx.year, idx.month
           if 'energy' not in row.keys():
               ndays[-1] = ndays[-1] + 1
               cdd[-1] = cdd[-1] + np.maximum(row['tempF'] - self.cooling_base_temp, 0)
               hdd[-1] = hdd[-1] + np.maximum(self.heating_base_temp - row['tempF'], 0)
           elif np.isfinite(row['energy']):
               ndays[-1] = ndays[-1] + 1
               usage[-1] = usage[-1] + row['energy']
               cdd[-1] = cdd[-1] + np.maximum(row['tempF'] - self.cooling_base_temp, 0)
               hdd[-1] = hdd[-1] + np.maximum(self.heating_base_temp - row['tempF'], 0)
        for i in range(len(usage)):
            if ndays[i] < 25:
                upd[i] = np.nan
                if ndays[i] == 0: cdd[i], hdd[i] = 0, 0
                else: cdd[i], hdd[i] = cdd[i]/ndays[i], hdd[i]/ndays[i]
            else:
                upd[i] = (usage[i] / ndays[i])
                cdd[i], hdd[i] = cdd[i]/ndays[i], hdd[i]/ndays[i]
        output = pd.DataFrame({'CDD': cdd, 'HDD': hdd, 'upd': upd, 'usage': usage}, index=output_index)
        if np.isnan(output['upd'][0]): output=output[1:]
        if np.isnan(output['upd'][-1]): output=output[:-1]
        return output

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

        data = pd.DataFrame(
            {'energy': trace_data.iloc[:-1], 'tempF': temperature_data},
            columns=['energy', 'tempF'])
        model_data = self._make_monthly(data)

        if np.sum(!np.isnan(df['upd'])) <= 12:
            return None

        # Fit the intercept-only model
        formula = 'upd ~ 1'
        try:
            int_mod = smf.ols(formula=formula, data=df)
            int_res = int_mod.fit()
            int_rsquared = int_res.rsquared
            int_qualified = (int_res.params['Intercept'] >= 0) and \
                            (int_res.pvalues['Intercept'] < 0.1)
        except:
            int_rsquared, int_qualified = 0, False
    
        # CDD-only
        formula = 'upd ~ CDD'
        try:
            if self.heating_only: assert False
            cdd_mod = smf.ols(formula=formula, data=df)
            cdd_res = cdd_mod.fit()
            cdd_rsquared = cdd_res.rsquared
            cdd_qualified = (cdd_res.params['Intercept'] >= 0) and (cdd_res.params['CDD'] >= 0) and \
                            (cdd_res.pvalues['Intercept'] < 0.1) and (cdd_res.pvalues['CDD'] < 0.1)
        except:
            cdd_rsquared, cdd_qualified = 0, False
    
        # HDD-only
        formula = 'upd ~ HDD'
        try:
            hdd_mod = smf.ols(formula=formula, data=df)
            hdd_res = hdd_mod.fit()
            hdd_rsquared = hdd_res.rsquared
            hdd_qualified = (hdd_res.params['Intercept'] >= 0) and (hdd_res.params['HDD'] >= 0) and \
                            (hdd_res.pvalues['Intercept'] < 0.1) and (hdd_res.pvalues['HDD'] < 0.1)
        except:
            hdd_rsquared, hdd_qualified = 0, False
    
        # CDD+HDD
        formula = 'upd ~ CDD + HDD'
        try:
            if self.heating_only: assert False
            full_mod = smf.ols(formula=formula, data=df)
            full_res = full_mod.fit()
            full_rsquared = full_res.rsquared
            full_qualified = (full_res.params['Intercept'] >= 0) and (full_res.params['HDD'] >= 0) and (full_res.params['CDD'] >= 0) and \
                             (full_res.pvalues['Intercept'] < 0.1) and (full_res.pvalues['CDD'] < 0.1) and (full_res.pvalues['HDD'] < 0.1)
        except:
            full_rsquared, full_qualified = 0, False
    
        if (full_qualified or hdd_qualified or cdd_qualified or int_qualified) == False: return None

        self.result = None
        if full_qualified and full_rsquared > \
          max([int(hdd_qualified)*hdd_rsquared,int(cdd_qualified)*cdd_rsquared,int(int_qualified)*int_rsquared]):
            self.formula = 'upd ~ CDD + HDD'
            self.result = full_res
        elif hdd_qualified and hdd_rsquared > \
          max([int(full_qualified)*full_rsquared,int(cdd_qualified)*cdd_rsquared,int(int_qualified)*int_rsquared]):
            self.formula = 'upd ~ HDD'
            self.result = hdd_res
        elif cdd_qualified and cdd_rsquared > \
          max([int(full_qualified)*full_rsquared,int(hdd_qualified)*hdd_rsquared,int(int_qualified)*int_rsquared]):
            self.formula = 'upd ~ CDD'
            self.result = cdd_res
        else:
            self.formula = 'upd ~ 1'
            self.result = int_res
    
        if self.result is None: return None

        estimated = self.result.predict(model_data)
        estimated = pd.Series(estimated, index=model_data.energy.index)
        for i in range(len(estimated.index)):
            estimated[i] = estimated[i] * \
                calendar.monthrange(estimated.index[i].year,estimated.index[i].month)[1]

        y,X = patsy.dmatrices(self.formula, model_data, return_type='dataframe')
        self.X = X
        for i in range(len(y.index)):
            y[i] = y[i] * calendar.monthrange(estimated.index[i].year,estimated.index[i].month)[1]
        self.y = y
        self.estimated = estimated

        r2 = self.result.rsquared
        rmse = ((y.values.ravel() - estimated.values.ravel())**2).mean()**.5

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
            "coefficients": self.result.params,
            "intercept": self.result.params['Intercept'],
            "X_design_info": X,
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

        data = demand_fixture_data.resample(
            pd.tseries.frequencies.Day()).agg({'tempF': np.mean})
        model_data = self._make_monthly(data)

        predicted = self.result.predict(model_data)
        predicted = pd.Series(predicted, index=model_data.index)
        for i in range(len(predicted.index)):
            predicted[i] = predicted[i]* calendar.monthrange(predicted.index[i].year,predicted.index[i].month)[1]

        return predicted

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
