import copy
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import eemeter.modeling.exceptions as model_exceptions
from eemeter.modeling.models.caltrack_helpers import \
    _fit_intercept, _fit_cdd_only, _fit_hdd_only, _fit_full


class CaltrackDailyModel(object):
    ''' This class implements the two-stage modeling routine agreed upon
    as part of the Caltrack beta test.

    If fit_cdd is True, then all four candidate models (HDD+CDD,
    CDD-only, HDD-only, and Intercept-only) are
    used in stage 1 estimation. If it's false, then only HDD-only and
    Intercept-only are used.

    If grid_search is set to True, the balance point temperatures are
    determined by maximizing R^2. Otherwise,
    70 and 60 degF are used for cooling and heating, respectively.

    Min_contiguous_months sets the number of contiguous months of data
    required at the beginning of the reporting period/end of the baseline
    period in order for the weather normalization to be valid.
    '''
    def __init__(
            self, fit_cdd=True, grid_search=False, min_fraction_coverage=0.9,
            min_contiguous_months=12,
            modeling_period_interpretation='baseline'):

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
        self.fit_bp_hdd, self.fit_bp_cdd = None, None
        self.min_fraction_coverage = min_fraction_coverage
        self.min_contiguous_months = min_contiguous_months
        self.modeling_period_interpretation = modeling_period_interpretation

        if grid_search:
            self.bp_cdd = range(65,76)
            self.bp_hdd = range(55,66)
        else:
            self.bp_cdd, self.bp_hdd = [70,], [60,]

    def __repr__(self):
        return 'CaltrackDailyModel'

    def ami_to_daily(self, df):
        ''' Convert from daily usage and temperature to monthly
        usage per day and average HDD/CDD. '''

        # Throw out any duplicate indices
        df = df[~df.index.duplicated(keep='last')].sort_index()

        # Create arrays to hold computed CDD and HDD for each
        # balance point temperature.
        cdd = {i: [0] for i in self.bp_cdd}
        hdd = {i: [0] for i in self.bp_hdd}

        # If there isn't any data, throw an exception
        if len(df.index) == 0:
            raise model_exceptions.DataSufficiencyException("No energy trace data")

        # Check whether we are creating a demand fixture.
        is_demand_fixture = 'energy' not in df.columns

        for bp in self.bp_cdd:
            cdd[bp] = pd.Series(
                np.maximum(df.tempF - bp, 0),
                index = df.index)
        for bp in self.bp_hdd:
            hdd[bp] = pd.Series(
                np.maximum(bp - df.tempF, 0),
                index = df.index)

        # spread out over the month
        ndays = pd.Series((is_demand_fixture or np.isfinite(df.energy)) &
                np.isfinite(hdd[self.bp_hdd[0]]),
                dtype=int)

        # Create output data frame
        if not is_demand_fixture:
            df_dict = {'upd': df.energy, 'usage': df.energy, 'ndays': ndays}
        else:
            df_dict = {'upd': ndays*0, 'usage': ndays*0, 'ndays': ndays}
        df_dict.update({'CDD_' + str(bp): cdd[bp] for bp in cdd.keys()})
        df_dict.update({'HDD_' + str(bp): hdd[bp] for bp in hdd.keys()})
        output = pd.DataFrame(df_dict, index=df.index)
        return output

    def meets_sufficiency_or_error(self, df):
        if np.sum(np.isfinite(df['usage'])) < self.min_fraction_coverage * len(df):
            raise model_exceptions.DataSufficiencyException("Insufficient coverage")
        if len(df) < self.min_contiguous_months * 30:
            raise model_exceptions.DataSufficiencyException("Insufficient data")
        return

    def fit(self, input_data):

        self.input_data = input_data
        if isinstance(input_data, tuple):
            raise model_exceptions.DataSufficiencyException(\
                  "Billing data is not appropriate for this model")
        else:
            df = self.ami_to_daily(self.input_data)
        self.df = df

        self.meets_sufficiency_or_error(df)

        # Fit the intercept-only model
        (
            int_formula,
            int_mod,
            int_res,
            int_rsquared,
            int_qualified
        ) = _fit_intercept(df)

        # CDD-only
        if self.fit_cdd:
            (
                cdd_formula,
                cdd_mod,
                cdd_res,
                cdd_rsquared,
                cdd_qualified,
                cdd_bp
            ) = _fit_cdd_only(df)
        else:
            cdd_formula = None
            cdd_mod = None
            cdd_res = None
            cdd_rsquared = 0
            cdd_qualified = False
            cdd_bp = None

        # HDD-only
        (
            hdd_formula,
            hdd_mod,
            hdd_res,
            hdd_rsquared,
            hdd_qualified,
            hdd_bp
        ) = _fit_hdd_only(df)

        # CDD+HDD
        if self.fit_cdd:
            (
                full_formula,
                full_mod,
                full_res,
                full_rsquared,
                full_qualified,
                full_hdd_bp,
                full_cdd_bp
            ) = _fit_full(df)
        else:
            full_formula = None
            full_mod = None
            full_res = None
            full_rsquared = 0
            full_qualified = False
            full_hdd_bp = None
            full_cdd_bp = None

        # Now we take the best qualified model.
        if (full_qualified or
            hdd_qualified or
            cdd_qualified or
            int_qualified) is False:
            raise model_exceptions.ModelFitException(
                "No candidate model fit to data successfully")

        use_full = (full_qualified and (
            full_rsquared > max([
                int(hdd_qualified) * hdd_rsquared,
                int(cdd_qualified) * cdd_rsquared,
                int(int_qualified) * int_rsquared,
            ])
        ))

        use_hdd_only = (hdd_qualified and (
            hdd_rsquared > max([
                int(full_qualified) * full_rsquared,
                int(cdd_qualified) * cdd_rsquared,
                int(int_qualified) * int_rsquared,
            ])
        ))

        use_cdd_only = (cdd_qualified and (
            cdd_rsquared > max([
                int(full_qualified) * full_rsquared,
                int(hdd_qualified) * hdd_rsquared,
                int(int_qualified) * int_rsquared,
            ])
        ))

        fit_bp_hdd, fit_bp_cdd = None, None

        if use_full:
            # Use the full model
            y, X = patsy.dmatrices(
                full_formula, df, return_type='dataframe')
            estimated = full_res.fittedvalues
            r2, rmse = full_rsquared, np.sqrt(full_res.ssr/full_res.nobs)
            model_obj, model_res, formula = full_mod, full_res, full_formula
            fit_bp_hdd, fit_bp_cdd = full_hdd_bp, full_cdd_bp

        elif use_hdd_only:
            y, X = patsy.dmatrices(
                hdd_formula, df, return_type='dataframe')
            estimated = hdd_res.fittedvalues
            r2, rmse = hdd_rsquared, np.sqrt(hdd_res.ssr/hdd_res.nobs)
            model_obj, model_res, formula = hdd_mod, hdd_res, hdd_formula
            fit_bp_hdd = hdd_bp

        elif use_cdd_only:
            y, X = patsy.dmatrices(
                cdd_formula, df, return_type='dataframe')
            estimated = cdd_res.fittedvalues
            r2, rmse = cdd_rsquared, np.sqrt(cdd_res.ssr/cdd_res.nobs)
            model_obj, model_res, formula = cdd_mod, cdd_res, cdd_formula
            fit_bp_cdd = cdd_bp

        else:
            # Use Intercept-only
            y, X = patsy.dmatrices(
                int_formula, df, return_type='dataframe')
            estimated = int_res.fittedvalues
            r2, rmse = int_rsquared, np.sqrt(int_res.ssr/int_res.nobs)
            model_obj, model_res, formula = int_mod, int_res, int_formula

        if y.mean != 0:
            cvrmse = rmse / float(y.values.ravel().mean())
            nmbe = np.nanmean(model_res.resid) / float(y.values.ravel().mean())
        else:
            cvrmse = np.nan
            nmbe = np.nan

        n = estimated.shape[0]

        self.y, self.X = y, X
        self.estimated = estimated
        self.r2, self.rmse = r2, rmse
        self.model_obj, self.model_res, self.formula = model_obj, model_res, formula
        self.cvrmse = cvrmse
        self.nmbe = nmbe
        self.fit_bp_hdd, self.fit_bp_cdd = fit_bp_hdd, fit_bp_cdd
        self.n = n
        self.params = {
            "coefficients": self.model_res.params.to_dict(),
            "formula": self.formula,
            "cdd_bp": self.fit_bp_cdd,
            "hdd_bp": self.fit_bp_hdd,
            "X_design_info": self.X.design_info,
        }

        output = {
            "r2": self.r2,
            "model_params": self.params,
            "rmse": self.rmse,
            "cvrmse": self.cvrmse,
            "nmbe": self.nmbe,
            "n": self.n,
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

        dfd = self.ami_to_daily(demand_fixture_data)

        formula = params["formula"]

        _, X = patsy.dmatrices(formula, dfd,
                               return_type='dataframe')

        try:
            predicted = self.model_res.predict(X)
            predicted = pd.Series(predicted, index=dfd.index)
            # Get parameter covariance matrix
            cov = self.model_res.cov_params()
            # Get prediction errors for each data point
            prediction_var = self.model_res.mse_resid + \
                (X * np.dot(cov, X.T).T).sum(1)
            predicted_baseline_use, predicted_baseline_use_var = 0.0, 0.0
        except:
            raise model_exceptions.ModelPredictException(
                "Prediction failed!")

        if summed:
            predicted = predicted.sum()
            variance = prediction_var.sum()
        else:
            output_data = pd.DataFrame({
                'predicted': predicted,
                'variance': prediction_var},
                index = predicted.index)
            predicted = output_data['predicted']
            variance = output_data['variance']
        return predicted, variance
