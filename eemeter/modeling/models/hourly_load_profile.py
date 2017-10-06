import pandas as pd
import datetime
import eemeter.modeling.exceptions as model_exceptions
from eemeter.modeling.models.caltrack_daily import CaltrackDailyModel


class HourlyLoadProfileModel(object):
    def __init__(
            self, fit_cdd=True, grid_search=False, min_fraction_coverage=0.9,
            min_contiguous_months=12,
            modeling_period_interpretation='baseline',
            **kwargs):

        self.fit_cdd = fit_cdd
        self.grid_search = grid_search
        self.model_freq = pd.tseries.frequencies.Day()
        self.params = None
        self.X = None
        self.y = None
        self.estimated = None
        self.r2 = None
        self.rmse = None
        self.cvrmse = None
        self.nmbe = None
        self.n = None
        self.input_data = None
        self.min_fraction_coverage = min_fraction_coverage
        self.min_contiguous_months = min_contiguous_months
        self.modeling_period_interpretation = modeling_period_interpretation
        self.caltrack_model = CaltrackDailyModel(
            fit_cdd=fit_cdd, grid_search=grid_search,
            min_fraction_coverage=min_fraction_coverage,
            min_contiguous_months=min_contiguous_months,
            modeling_period_interpretation=modeling_period_interpretation)

    def __repr__(self):
        return 'HourlyLoadProfileModel'

    def fit(self, input_data):
        if isinstance(input_data, tuple):
            raise model_exceptions.DataSufficiencyException(
                  "Billing data is not appropriate for this model")
        self.input_data = input_data
        input_data_daily = input_data.resample('D').apply(
            {'energy': pd.Series.sum, 'tempF': pd.Series.mean})
        self.caltrack_model.fit(input_data_daily)

        self.params = {
            "coefficients": self.caltrack_model.model_res.params.to_dict(),
            "formula": self.caltrack_model.formula,
            "cdd_bp": self.caltrack_model.fit_bp_cdd,
            "hdd_bp": self.caltrack_model.fit_bp_hdd,
            "X_design_info": self.caltrack_model.X.design_info,
        }

        output = {
            "r2": self.caltrack_model.r2,
            "model_params": self.caltrack_model.params,
            "rmse": self.caltrack_model.rmse,
            "cvrmse": self.caltrack_model.cvrmse,
            "nmbe": self.caltrack_model.nmbe,
            "n": self.caltrack_model.n,
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

        df_daily, _ = self.caltrack_model.predict(
            demand_fixture_data.resample('D').mean(),
            summed=False)
        output_data = pd.DataFrame({
            'predicted': demand_fixture_data.tempF * 0.,
            'variance': demand_fixture_data.tempF * 0.},
            index=demand_fixture_data.index)

        ii = self.input_data.groupby([self.input_data.index.month,
                                      self.input_data.index.dayofweek < 5,
                                      self.input_data.index.hour
                                      ]).energy.mean()
        jj = self.input_data.groupby([self.input_data.index.month,
                                      self.input_data.index.dayofweek < 5,
                                      self.input_data.index.hour]).energy.std()
        output_data.predicted = ii.loc[list(zip(output_data.index.month,
                                                output_data.index.dayofweek < 5,
                                                output_data.index.hour))].values
        output_data.variance = jj.loc[list(zip(output_data.index.month,
                                               output_data.index.dayofweek < 5,
                                               output_data.index.hour))].values
        output_data_daily = output_data.predicted.resample('D').sum()
        output_factors = df_daily / output_data_daily
        nextday = df_daily.index[-1] + datetime.timedelta(days=1)
        output_factors = output_factors.append(pd.Series([0.],
                                               index=pd.date_range(nextday,
                                                                   nextday)))
        output_factors = output_factors.resample('H').ffill()[:-1]
        output_data['predicted'] = output_data['predicted'] * output_factors
        output_data['variance'] = output_data['variance'] * output_factors**2

        if summed:
            predicted = output_data['predicted'].sum()
            variance = output_data['variance'].sum()
        else:
            predicted = output_data['predicted']
            variance = output_data['variance']
        return predicted, variance
