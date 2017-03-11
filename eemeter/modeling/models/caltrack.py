import copy
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
import eemeter.modeling.exceptions as model_exceptions


class CaltrackMonthlyModel(object):
    ''' This class implements the two-stage modeling routine agreed upon
    as part of the Caltrack beta test.

    If fit_cdd is True, then all four candidate models (HDD+CDD,
    CDD-only, HDD-only, and Intercept-only) are
    used in stage 1 estimation. If it's false, then only HDD-only and
    Intercept-only are used.

    If grid_search is set to True, the balance point temperatures are
    determined by maximizing R^2 across the range 50-85 degF. Otherwise,
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
        self.fit_bp_hdd, self.fit_bp_cdd = None, None
        self.min_contiguous_months = min_contiguous_months
        self.modeling_period_interpretation = modeling_period_interpretation

        if grid_search:
            self.bp_cdd = [50, 55, 60, 65, 70, 75, 80, 85]
            self.bp_hdd = [50, 55, 60, 65, 70, 75, 80, 85]
        else:
            self.bp_cdd, self.bp_hdd = [70,], [60,]

    def __repr__(self):
        return 'CaltrackMonthlyModel'

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
        temp_data = temp_data.resample('D').apply(np.mean)[0]

        # Drop any duplicate indices
        energy_data = energy_data[
            ~energy_data.index.duplicated(keep='last')].sort_index()

        # Check for empty series post-resampling and deduplication
        if energy_data.empty:
            raise model_exceptions.DataSufficiencyException(
                "No energy trace data after deduplication")
        if temp_data.empty:
            raise model_exceptions.DataSufficiencyException(
                "No temperature data after resampling")

        # get daily mean values
        energy_data_daily_mean_values = [
            value / (e - s).days for value, s, e in
            zip(energy_data, energy_data.index, energy_data.index[1:])
        ] + [np.nan]  # add missing last data point, which is null by convention anyhow

        # spread out over the month
        energy_data_daily = pd.Series(
            energy_data_daily_mean_values,
            index=energy_data.index
            ).resample('D').ffill()[:-1]

        model_data = pd.DataFrame({
            'energy': energy_data_daily,
            'tempF': temp_data,
        })

        return model_data

    def daily_to_monthly_avg(self, df):
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

        # Create the arrays to hold our monthly output
        ndays, usage, upd, output_index = [0], [0], [0], [df.index[0]]
        this_yr, this_mo = output_index[0].year, output_index[0].month

        # Check whether we are creating a demand fixture.
        is_demand_fixture = 'energy' not in df.columns

        # TODO use groupby here? e.g. df.groupby(pd.TimeGrouper('MS'))
        # Loop through the daily input data frame populating monthly arrays
        for idx, row in df.iterrows():
            # Check whether we are in a new month.
            new_month = (this_yr != idx.year or this_mo != idx.month)
            if new_month:
                ndays.append(0)
                usage.append(0)
                upd.append(0)
                for i in cdd.keys():
                    cdd[i].append(0)
                for i in hdd.keys():
                    hdd[i].append(0)
                this_yr, this_mo = idx.year, idx.month
                output_index.append(idx)

            # If this day is valid, add it to the usage and CDD/HDD arrays.
            day_is_valid = (
                (is_demand_fixture or np.isfinite(row['energy'])) and
                np.isfinite(row['tempF']))
            if day_is_valid:
                ndays[-1] = ndays[-1] + 1
                usage[-1] = usage[-1] + (
                    row['energy'] if not is_demand_fixture else 0)
                for bp in cdd.keys():
                    cdd[bp][-1] += np.maximum(row['tempF'] - bp, 0)
                for bp in hdd.keys():
                    hdd[bp][-1] += np.maximum(bp - row['tempF'], 0)

        # Caltrack sufficiency requirement of >=15 days per month
        for i in range(len(usage)):
            misses_req = (ndays[i] < 15)
            if misses_req:
                upd[i] = np.nan
                for bp in cdd.keys():
                    cdd[bp][i] = np.nan
                for bp in hdd.keys():
                    hdd[bp][i] = np.nan
            else:
                upd[i] = (usage[i] / ndays[i])
                for bp in cdd.keys():
                    cdd[bp][i] = cdd[bp][i] / ndays[i]
                for bp in hdd.keys():
                    hdd[bp][i] = hdd[bp][i] / ndays[i]

        # Create output data frame
        df_dict = {'upd': upd, 'usage': usage, 'ndays': ndays}
        df_dict.update({'CDD_' + str(bp): cdd[bp] for bp in cdd.keys()})
        df_dict.update({'HDD_' + str(bp): hdd[bp] for bp in hdd.keys()})
        output = pd.DataFrame(df_dict, index=output_index)
        return output

    def monthly_avg_to_daily(self, input_data, index=None):
        if index is None:
            index = pd.date_range(
                input_data.index[0],
                input_data.index[-1].to_period('M').to_timestamp('M'),
                freq='d')
        output_data = input_data.reindex(index, method='ffill')
        if 'usage' in output_data.columns:
            del output_data['usage']
        if 'ndays' in output_data.columns:
            del output_data['ndays']
        return output_data

    def meets_sufficiency_or_error(self, df):
        # Caltrack sufficiency requirement of number of contiguous months
        _n = self.min_contiguous_months

        # choose first hdd as a proxy for temperature data
        upd = df['upd'].values
        hdd_col = [col for col in df.columns if col.startswith('HDD')][0]
        temp = df[hdd_col].values

        def n_non_nan(values):
            return np.sum(~np.isnan(values))

        reason = None
        mp_type = self.modeling_period_interpretation
        if mp_type == 'baseline':

            # In the baseline period, require the last N months be non-nan.
            last_month_nan = np.isnan(upd[-1])
            direction = "last"

            if last_month_nan:
                upd_contig = upd[-(_n+1):-1]
                temp_contig = temp[-(_n+1):-1]
            else:
                upd_contig = upd[-_n:]
                temp_contig = temp[-_n:]

        elif mp_type == 'reporting':

            # In the reporting period, require the first N months be non-nan.
            first_month_nan = np.isnan(df['upd'].values[0])
            direction = "first"

            if first_month_nan:
                upd_contig = upd[1:_n+1]
                temp_contig = temp[1:_n+1]
            else:
                upd_contig = upd[:_n]
                temp_contig = temp[:_n]
        else:
            raise ValueError(
                'Unexpected modeling period interpretation {}'
                .format(mp_type)
            )

        n_months = len(upd_contig)
        if n_months < _n:
            reason = (
                'The {direction} {req} months of a {mp} period must have'
                ' non-NaN energy and temperature values. In this case, there'
                ' were only {n} months in the series.'
                .format(
                    direction=direction,
                    req=_n,
                    mp=mp_type,
                    n=n_months
                )
            )
        else:
            upd_n_non_nan = n_non_nan(upd_contig)
            temp_n_non_nan = n_non_nan(temp_contig)
            upd_ok = (upd_n_non_nan == _n)
            temp_ok = (temp_n_non_nan == _n)
            if upd_ok and not temp_ok:
                reason = (
                    'The {direction} {req} months of a {mp} period must have'
                    ' at least 15 valid days of energy and temperature data.'
                    ' In this case, only {n} of the {direction} {req} months'
                    ' of temperature data met that requirement.'
                    .format(
                        direction=direction,
                        req=_n,
                        mp=mp_type,
                        n=temp_n_non_nan,
                    )
                )
            elif not upd_ok and temp_ok:
                reason = (
                    'The {direction} {req} months of a {mp} period must have'
                    ' at least 15 valid days of energy and temperature data.'
                    ' In this case, only {n} of the {direction} {req} months'
                    ' of energy data met that requirement.'
                    .format(
                        direction=direction,
                        req=_n,
                        mp=mp_type,
                        n=upd_n_non_nan,
                    )
                )
            elif not upd_ok and not temp_ok:
                reason = (
                    'The {direction} {req} months of a {mp} period must have'
                    ' at least 15 valid days of energy and temperature data.'
                    ' In this case, only {upd_n} and {temp_n} of the'
                    ' {direction} {req} months of energy and temperature data'
                    ' met that requirement, respectively.'
                    .format(
                        direction=direction,
                        req=_n,
                        mp=mp_type,
                        upd_n=upd_n_non_nan,
                        temp_n=temp_n_non_nan
                    )
                )

        if reason is not None:
            raise model_exceptions.DataSufficiencyException(
                'Data does not meet minimum contiguous months requirement. {}'
                .format(reason)
            )

        if not np.nansum(upd) > 0.01:
            raise model_exceptions.DataSufficiencyException(
                "Energy trace data is all or nearly all zero")

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
                    full_formula = 'upd ~ CDD_' + full_cdd_bp + \
                                   ' + HDD_' + full_hdd_bp
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

        if isinstance(input_data, tuple):
            self.input_data = self.billing_to_daily(input_data)
        else:
            self.input_data = input_data

        df = self.daily_to_monthly_avg(self.input_data)

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
        else:
            cvrmse = np.nan

        n = estimated.shape[0]

        self.y, self.X = y, X
        self.estimated = estimated
        self.r2, self.rmse = r2, rmse
        self.model_obj, self.model_res, self.formula = model_obj, model_res, formula
        self.cvrmse = cvrmse
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

        formula = params["formula"]

        demand_fixture_index = demand_fixture_data.index.copy()
        demand_fixture_data = self.daily_to_monthly_avg(demand_fixture_data)
        dfd = demand_fixture_data.dropna()

        _, X = patsy.dmatrices(formula, dfd,
                               return_type='dataframe')

        try:
            predicted = self.model_res.predict(X)
            predicted = pd.Series(predicted, index=dfd.index)
            variance = copy.deepcopy(predicted)
            # Get parameter covariance matrix
            cov = self.model_res.cov_params()
            # Get prediction errors for each data point
            prediction_var = self.model_res.mse_resid + \
                (X * np.dot(cov, X.T).T).sum(1)
            predicted_baseline_use, predicted_baseline_use_var = 0.0, 0.0
        except:
            raise model_exceptions.ModelPredictException(
                "Prediction failed!")

        if not np.all(~np.isnan(predicted)):
            raise model_exceptions.ModelPredictException(
                "Prediction has NaN values")

        if not np.all(~np.isnan(prediction_var)):
            raise model_exceptions.ModelPredictException(
                "Prediction has NaN variances")

        if summed:
        # Sum them up using the number of days in the demand fixture.
            for i in demand_fixture_data.index:
                if i not in predicted.index or not np.isfinite(predicted[i]):
                    continue
                predicted[i] = predicted[i] * demand_fixture_data.ndays[i]
                predicted_baseline_use = predicted_baseline_use + predicted[i]
                variance[i] = prediction_var[i] * demand_fixture_data.ndays[i]
                predicted_baseline_use_var = \
                    predicted_baseline_use_var + variance[i]

            predicted = predicted_baseline_use
            variance = predicted_baseline_use_var
        else:
            input_data = pd.DataFrame({
                'predicted': predicted,
                'variance': prediction_var},
                index = predicted.index)
            output_data = self.monthly_avg_to_daily(input_data,
                index=demand_fixture_index)
            predicted = output_data['predicted']
            variance = output_data['variance']
        return predicted, variance
