import copy
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import eemeter.modeling.exceptions as model_exceptions

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
            self, fit_cdd=True, grid_search=False, min_contiguous_months=12,
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
        self.min_contiguous_months = min_contiguous_months
        self.modeling_period_interpretation = modeling_period_interpretation

        if grid_search:
            self.bp_cdd = range(65,76)
            self.bp_hdd = range(55,66)
        else:
            self.bp_cdd, self.bp_hdd = [70,], [60,]

    def __repr__(self):
        return 'CaltrackDailyModel'

    def billing_to_daily(self, trace_and_temp):
        ''' Helper function to handle monthly billing or other irregular data.
        '''
        (energy_data, temp_data) = trace_and_temp

        # Handle empty series
        if energy_data.empty:
            raise model_exceptions.DataSufficiencyException("No energy trace data")
        if temp_data.empty:
            raise model_exceptions.DataSufficiencyException("No temperature data")

        # Convert billing multiindex to straight index
        temp_data.index = temp_data.index.droplevel()

        # Resample temperature data to daily
        temp_data_daily = temp_data.resample('D').apply(np.mean)[0]

        # Drop any duplicate indices
        energy_data = energy_data[
            ~energy_data.index.duplicated(keep='last')].sort_index()

        # Check for empty series post-resampling and deduplication
        if energy_data.empty:
            raise model_exceptions.DataSufficiencyException(
                "No energy trace data after deduplication")
        if temp_data_daily.empty:
            raise model_exceptions.DataSufficiencyException(
                "No temperature data after resampling")

        # get daily mean values
        upd_data_daily_mean_values = [
            value / (e - s).days for value, s, e in
            zip(energy_data, energy_data.index, energy_data.index[1:])
        ] + [np.nan]  # add missing last data point, which is null by convention anyhow
        usage_data_daily_mean_values = [
            value for value, s, e in
            zip(energy_data, energy_data.index, energy_data.index[1:])
        ] + [np.nan]  # add missing last data point, which is null by convention anyhow

        # Create arrays to hold computed CDD and HDD for each
        # balance point temperature.
        cdd = {i: [0] for i in self.bp_cdd}
        hdd = {i: [0] for i in self.bp_hdd}
        for bp in self.bp_cdd:
            cdd[bp] = pd.Series(
                np.maximum(temp_data_daily - bp, 0),
                index = temp_data_daily.index)
        for bp in self.bp_hdd:
            hdd[bp] = pd.Series(
                np.maximum(bp - temp_data_daily, 0),
                index = temp_data_daily.index)

        # spread out over the month
        upd_data = pd.Series(
            upd_data_daily_mean_values,
            index=energy_data.index
            ).resample('D').ffill()[:-1] 
        usage_data = pd.Series(
            usage_data_daily_mean_values,
            index=energy_data.index
            ).resample('D').ffill()[:-1] 
        cdd_data = {}
        hdd_data = {}
        for bp in self.bp_cdd:
            cdd_data[bp] = pd.Series(
                cdd[bp] + [np.nan],
                index=energy_data.index
                ).resample('D').ffill()[:-1] 
        for bp in self.bp_hdd:
            hdd_data[bp] = pd.Series(
                hdd[bp] + [np.nan],
                index=energy_data.index
                ).resample('D').ffill()[:-1] 
        ndays_data = pd.Series(np.isfinite(upd_data) &
                     np.isfinite(hdd_data[self.bp_hdd[0]]),
                     dtype=int)

        model_data = {
            'upd': upd_data,
            'usage': usage_data,
            'ndays': ndays_data,
        }
        model_data.update({'CDD_' + str(bp): \
            cdd_data[bp] for bp in cdd_data.keys()})
        model_data.update({'HDD_' + str(bp): \
            hdd_data[bp] for bp in hdd_data.keys()})

        return pd.DataFrame(model_data)

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
        # XXX Put in criteria
        return

    def _fit_intercept(self, df):
        int_formula = 'upd ~ 1'
        try:
            int_mod = smf.ols(formula=int_formula, data=df)
            int_res = int_mod.fit()
        except:  # TODO: catch specific error
            int_rsquared, int_qualified = 0, False
            int_formula, int_mod, int_res = None, None, None
        else:
            int_rsquared, int_qualified = 0, True

        return int_formula, int_mod, int_res, int_rsquared, int_qualified

    def _fit_cdd_only(self, df):

        bps = [i[4:] for i in df.columns if i[:3] == 'CDD']
        best_bp, best_rsquared, best_mod, best_res = None, -9e9, None, None

        try:  # TODO: fix big try block anti-pattern
            for bp in bps:
                cdd_formula = 'upd ~ CDD_' + bp
                if (np.nansum(df['CDD_' + bp] > 0) < 10) or \
                   (np.nansum(df['CDD_' + bp]) < 20):
                    continue
                cdd_mod = smf.ols(formula=cdd_formula, data=df)
                cdd_res = cdd_mod.fit()
                cdd_rsquared = cdd_res.rsquared
                if (cdd_rsquared > best_rsquared and
                        cdd_res.params['Intercept'] >= 0 and
                        cdd_res.params['CDD_' + bp] >= 0):
                    best_bp, best_rsquared = bp, cdd_rsquared
                    best_mod, best_res = cdd_mod, cdd_res
            if (best_bp is not None and
                    (best_res.pvalues['Intercept'] < 0.1) and
                    (best_res.pvalues['CDD_' + best_bp] < 0.1)):
                cdd_qualified = True
                cdd_formula = 'upd ~ CDD_' + best_bp
                cdd_bp = int(best_bp)
                cdd_mod, cdd_res, cdd_rsquared = \
                    best_mod, best_res, best_rsquared
            else:
                cdd_rsquared, cdd_qualified = 0, False
                cdd_formula, cdd_mod, cdd_res = None, None, None
                cdd_bp = None
        except:  # TODO: catch specific error
            cdd_rsquared, cdd_qualified = 0, False
            cdd_formula, cdd_mod, cdd_res = None, None, None
            cdd_bp = None

        return cdd_formula, cdd_mod, cdd_res, cdd_rsquared, cdd_qualified, cdd_bp

    def _fit_hdd_only(self, df):

        bps = [i[4:] for i in df.columns if i[:3] == 'HDD']
        best_bp, best_rsquared, best_mod, best_res = None, -9e9, None, None

        try:  # TODO: fix big try block anti-pattern
            for bp in bps:
                hdd_formula = 'upd ~ HDD_' + bp
                if (np.nansum(df['HDD_' + bp] > 0) < 10) or \
                   (np.nansum(df['HDD_' + bp]) < 20):
                    continue
                hdd_mod = smf.ols(formula=hdd_formula, data=df)
                hdd_res = hdd_mod.fit()
                hdd_rsquared = hdd_res.rsquared
                if (hdd_rsquared > best_rsquared and
                        hdd_res.params['Intercept'] >= 0 and
                        hdd_res.params['HDD_' + bp] >= 0):
                    best_bp, best_rsquared = bp, hdd_rsquared
                    best_mod, best_res = hdd_mod, hdd_res

            if (best_bp is not None and
                    (best_res.pvalues['Intercept'] < 0.1) and
                    (best_res.pvalues['HDD_' + best_bp] < 0.1)):
                hdd_qualified = True
                hdd_formula = 'upd ~ HDD_' + best_bp
                hdd_bp = int(best_bp)
                hdd_mod, hdd_res, hdd_rsquared = \
                    best_mod, best_res, best_rsquared
            else:
                hdd_rsquared, hdd_qualified = 0, False
                hdd_formula, hdd_mod, hdd_res = None, None, None
                hdd_bp = None
        except:  # TODO: catch specific error
            hdd_rsquared, hdd_qualified = 0, False
            hdd_formula, hdd_mod, hdd_res = None, None, None
            hdd_bp = None

        return hdd_formula, hdd_mod, hdd_res, hdd_rsquared, hdd_qualified, hdd_bp

    def _fit_full(self, df):

        hdd_bps = [i[4:] for i in df.columns if i[:3] == 'HDD']
        cdd_bps = [i[4:] for i in df.columns if i[:3] == 'CDD']

        best_hdd_bp, best_cdd_bp, best_rsquared, best_mod, best_res = \
            None, None, -9e9, None, None

        try:  # TODO: fix big try block anti-pattern
            for full_hdd_bp in hdd_bps:
                for full_cdd_bp in cdd_bps:
                    if full_cdd_bp < full_hdd_bp: continue
                    full_formula = 'upd ~ CDD_' + full_cdd_bp + \
                                   ' + HDD_' + full_hdd_bp
                    if (np.nansum(df['HDD_' + full_hdd_bp] > 0) < 10) or \
                       (np.nansum(df['HDD_' + full_hdd_bp]) < 20):
                        continue
                    if (np.nansum(df['CDD_' + full_cdd_bp] > 0) < 10) or \
                       (np.nansum(df['CDD_' + full_cdd_bp]) < 20):
                        continue
                    full_mod = smf.ols(formula=full_formula, data=df)
                    full_res = full_mod.fit()
                    full_rsquared = full_res.rsquared
                    if (full_rsquared > best_rsquared and
                            full_res.params['Intercept'] >= 0 and
                            full_res.params['HDD_' + full_hdd_bp] >= 0 and
                            full_res.params['CDD_' + full_cdd_bp] >= 0):
                        best_hdd_bp, best_cdd_bp, best_rsquared = \
                            full_hdd_bp, full_cdd_bp, full_rsquared
                        best_mod, best_res = full_mod, full_res

            if (best_hdd_bp is not None and
                    (best_res.pvalues['Intercept'] < 0.1) and
                    (best_res.pvalues['CDD_' + best_cdd_bp] < 0.1) and
                    (best_res.pvalues['HDD_' + best_hdd_bp] < 0.1)):
                full_qualified = True
                full_formula = 'upd ~ CDD_' + best_cdd_bp + \
                               ' + HDD_' + best_hdd_bp
                full_hdd_bp = int(best_hdd_bp)
                full_cdd_bp = int(best_cdd_bp)
                full_mod, full_res, full_rsquared = \
                    best_mod, best_res, best_rsquared
            else:
                full_rsquared, full_qualified = 0, False
                full_formula, full_mod, full_res = None, None, None
                full_hdd_bp, full_hdd_bp = None, None
        except:  # TODO: catch specific error
            full_rsquared, full_qualified = 0, False
            full_formula, full_mod, full_res = None, None, None
            full_hdd_bp, full_hdd_bp = None, None

        return full_formula, full_mod, full_res, full_rsquared, full_qualified, full_hdd_bp, full_cdd_bp


    def fit(self, input_data):

        self.input_data = input_data
        if isinstance(input_data, tuple):
            df = self.billing_to_daily(input_data)
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
        ) = self._fit_intercept(df)

        # CDD-only
        if self.fit_cdd:
            (
                cdd_formula,
                cdd_mod,
                cdd_res,
                cdd_rsquared,
                cdd_qualified,
                cdd_bp
            ) = self._fit_cdd_only(df)
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
        ) = self._fit_hdd_only(df)

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
            ) = self._fit_full(df)
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
            r2, rmse = full_rsquared, np.sqrt(full_res.mse_total)
            model_obj, model_res, formula = full_mod, full_res, full_formula
            fit_bp_hdd, fit_bp_cdd = full_hdd_bp, full_cdd_bp

        elif use_hdd_only:
            y, X = patsy.dmatrices(
                hdd_formula, df, return_type='dataframe')
            estimated = hdd_res.fittedvalues
            r2, rmse = hdd_rsquared, np.sqrt(hdd_res.mse_total)
            model_obj, model_res, formula = hdd_mod, hdd_res, hdd_formula
            fit_bp_hdd = hdd_bp

        elif use_cdd_only:
            y, X = patsy.dmatrices(
                cdd_formula, df, return_type='dataframe')
            estimated = cdd_res.fittedvalues
            r2, rmse = cdd_rsquared, np.sqrt(cdd_res.mse_total)
            model_obj, model_res, formula = cdd_mod, cdd_res, cdd_formula
            fit_bp_cdd = cdd_bp

        else:
            # Use Intercept-only
            y, X = patsy.dmatrices(
                int_formula, df, return_type='dataframe')
            estimated = int_res.fittedvalues
            r2, rmse = int_rsquared, np.sqrt(int_res.mse_total)
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
