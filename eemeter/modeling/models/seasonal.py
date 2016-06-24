import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.stats import chi2
import holidays
import patsy
import warnings


class SeasonalElasticNetCVModel(object):

    def __init__(self, cooling_base_temp, heating_base_temp):

        self.cooling_base_temp = cooling_base_temp
        self.heating_base_temp = heating_base_temp

        self.model_freq = pd.tseries.frequencies.Day()
        self.base_formula = 'energy ~ 1 + CDD + HDD + CDD:HDD'
        self.l1_ratio = 0.5
        self.holidays = holidays.UnitedStates()
        self.params = None
        self.X = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None
        self.n = None

    def fit(self, df):
        # convert to daily
        model_data = df.resample(self.model_freq).agg(
                {'energy': np.sum, 'tempF': np.mean})

        model_data = model_data.dropna()

        if model_data.empty:
            raise ValueError("No model data (consumption + weather)")

        model_data.loc[:, 'CDD'] = np.maximum(model_data.tempF -
                                              self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.maximum(self.heating_base_temp -
                                              model_data.tempF, 0.)

        formula = self.base_formula

        # Make sure these factors have enough levels to not
        # cause issues.
        if len(np.unique(model_data.index.month)) >= 12:
            formula += '''\
            + CDD * C(tempF.index.month) \
            + HDD * C(tempF.index.month) \
            + C(tempF.index.month) \
            '''

        if len(np.unique(model_data.index.weekday)) >= 7:
            formula += '''\
            + (CDD) * C(tempF.index.weekday) \
            + (HDD) * C(tempF.index.weekday) \
            + C(tempF.index.weekday)\
            '''

        holiday_names = pd.Series(map(lambda x:  self.holidays.get(x, "none"),
                                      model_data.index),
                                  index=model_data.index)

        if len(np.unique(holiday_names)) >= 13:
            model_data.loc[:, 'holiday_name'] = holiday_names
            formula += " + C(holiday_name)"

        y, X = patsy.dmatrices(formula, model_data, return_type='dataframe')

        model_obj = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio,
                                              fit_intercept=False)
        model_obj.fit(X, y.values.ravel())

        estimated = pd.Series(model_obj.predict(X),
                              index=model_data.tempF.index)

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
            "formula": formula,
        }
        return self.params, self.r2, self.cvrmse

    def predict(self, df, params=None):
        # needs only tempF
        if params is None:
            params = self.params

        model_data = df.resample(self.model_freq).agg({'tempF': np.mean})

        model_data.loc[:, 'CDD'] = np.maximum(model_data.tempF -
                                              self.cooling_base_temp, 0.)
        model_data.loc[:, 'HDD'] = np.maximum(self.heating_base_temp -
                                              model_data.tempF, 0.)

        holiday_names = pd.Series(map(lambda x:  self.holidays.get(x, "none"),
                                      model_data.index),
                                  index=model_data.index)

        model_data.loc[:, 'holiday_name'] = holiday_names

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

        return predicted

    def plot(self):

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
