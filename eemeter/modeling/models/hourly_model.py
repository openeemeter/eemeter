import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import patsy
import eemeter.modeling.exceptions as model_exceptions


class HourlyDayOfWeekModel(object):
    """
    A simple regression model based on "Day of Week", Temparature and hour
    of day features.
    Two separate linear regression models are created for--weeekdays and
    weekends.
    The fit function takes as input a dataframe indexed with hourly timestamps
    and tempF as column which contain hourly temparatures.
    """
    def __init__(self, cdd_base_temp=70, hdd_base_temp=60, fit_cdd=True, fit_hdd=True, grid_search=False,
                 min_fraction_coverage=0.9, min_contiguous_months=1, modeling_period_interpretation='baseline',
                 **kwargs):
        self.model_weekday = None
        self.model_res_weekday = None
        self.model_weekend = None
        self.model_res_weekend = None
        self.formula = 'energy ~ hdd + cdd +' \
                       'hour_of_day + day_of_week + hour_of_day:day_of_week'
        self.weekdays = ['0', '1', '2', '3', '4']
        self.weekends = ['5', '6']
        self.cdd_base_temp = cdd_base_temp
        self.hdd_base_temp = hdd_base_temp

        # Following attributes are not used but adding it here, so as to make it similar to other
        # Model initialization.
        self.fit_cdd = fit_cdd
        self.fit_hdff = fit_hdd
        self.grid_search = grid_search
        self.min_fraction_coverage = min_fraction_coverage,
        self.min_contiguous_months = min_contiguous_months
        self.modeling_period_interpretation = modeling_period_interpretation

    def add_time_day(self, df):
        """

        Parameters
        ----------
        df : DataFrame, indexed by hour.
        Returns
        -------
        A new datafarame with two more columns:
        hour_of_day and day_of_week
        """
        hour_of_day = []
        day_of_week = []
        for index, row in df.iterrows():
            hour_of_day.append(str(index.hour))
            day_of_week.append(str(index.dayofweek))

        new_df = df.assign(hour_of_day=hour_of_day,
                           day_of_week=day_of_week)

        return new_df

    def add_hdd(self, df):
        if 'tempF' not in df:
            raise ValueError('tempF column not in Dataframe')

        hdd = np.maximum(self.hdd_base_temp - df.tempF, 0)
        df_with_hdd = df.assign(hdd=hdd)
        return df_with_hdd

    def add_cdd(self, df):
        if 'tempF' not in df:
            raise ValueError('tempF column not in Dataframe')

        cdd = np.maximum(df.tempF - self.cdd_base_temp, 0)
        df_with_cdd = df.assign(cdd=cdd)
        return df_with_cdd

    def get_model_stats(self,
                        model_res, df):
        if not model_res:
            return {}
        rmse = np.sqrt(model_res.ssr/model_res.nobs)
        cvrmse = rmse / df['energy'].mean()
        nmbe = np.nanmean(model_res.resid) / df['energy'].mean()

        result = {'intercept': model_res.params['Intercept'],
                  'r2': model_res.rsquared_adj,
                  'rmse': rmse,
                  'cvrmse': cvrmse,
                  'nmbe': nmbe}
        return result

    def meets_sufficiency_or_error(self, df):
        if len(df) < self.min_contiguous_months * 30 * 24:
            raise model_exceptions.\
                DataSufficiencyException("Min Contigous Month criteria not satisifed: Min Months Reqd:  " +
                                         str(self.min_contiguous_months))

    def fit(self, df):
        """
        Parameters
        ----------
        df A dataframe indexed by hour as frequency and with tempF (hourly temp)
        and energy as column
        Fits two models: One considering only weekdays and another considering
        only weekends.
        Returns
        -------
        """
        train_df = self.add_time_day(df)
        self.meets_sufficiency_or_error(train_df)

        weekday_df = train_df.loc[train_df['day_of_week'].isin(self.weekdays)]
        weekday_df = self.add_cdd(weekday_df)
        weekday_df = self.add_hdd(weekday_df)

        weekend_df = train_df.loc[train_df['day_of_week'].isin(self.weekends)]
        weekend_df = self.add_hdd(weekend_df)
        weekend_df = self.add_cdd(weekend_df)

        try:
            self.model_weekday = smf.ols(formula=self.formula, data=weekday_df)
            self.model_res_weekday = self.model_weekday.fit()
        except ValueError:
            self.model_weekday = None
            self.model_res_weekday = None

        try:
            self.model_weekend = smf.ols(formula=self.formula, data=weekend_df)
            self.model_res_weekend = self.model_weekend.fit()
        except ValueError:
            self.model_weekend = None
            self.model_res_weekend = None

        params = {
            "coefficients": self.model_res_weekday.params.to_dict(),
            "coefficients_weekend": self.model_res_weekend.params.to_dict(),
            "formula": self.formula,
            "cdd_bp": self.cdd_base_temp,
            "hdd_bp": self.hdd_base_temp,
            "X_design_info": ''
        }

        weekday_model_stats = self.get_model_stats(self.model_res_weekday, weekday_df)
        weekend_model_stats = self.get_model_stats(self.model_res_weekend, weekend_df)
        weekend_r2 = weekend_model_stats.get('r2', np.nan)
        weekday_r2 = weekday_model_stats.get('r2', np.nan)
        r2 = np.nan
        if not np.isnan(weekday_r2) and not np.isnan(weekend_r2):
            r2 = (weekday_r2 + weekend_r2) / 2.0

        weekend_rmse = weekend_model_stats.get('rmse', np.nan)
        weekday_rmse = weekday_model_stats.get('rmse', np.nan)
        rmse = np.nan
        cvrmse = np.nan
        nmbe = np.nan
        if not np.isnan(weekday_rmse) and not np.isnan(weekend_rmse):
            rmse = (weekend_rmse + weekday_rmse) / 2.0
            cvrmse = (weekend_model_stats.get('cvrmse') + weekday_model_stats.get('cvrmse')) / 2.0
            nmbe = (weekend_model_stats.get('cvrmse') + weekday_model_stats.get('cvrmse')) / 2.0

        output = {
            'r2': r2,
            'model_params': params,
            'rmse': rmse,
            'cvrmse': cvrmse,
            'nmbe': nmbe,
            'weekday_rmse': weekday_rmse,
            'weekend_rmse': weekend_rmse,
            'n':  len(train_df)
        }
        return output

    def compute_variance(self, df):
        weekday_df = df.loc[df['day_of_week'].isin(self.weekdays)]
        weekend_df = df.loc[df['day_of_week'].isin(self.weekends)]

        _, weekday_X = patsy.dmatrices(self.formula,
                                       weekday_df,
                                       return_type='dataframe')

        cov = self.model_res_weekday.cov_params()

        weekday_var = self.model_res_weekday.mse_resid + (weekday_X * np.dot(cov, weekday_X.T).T).sum(1)

        _, weekend_X = patsy.dmatrices(self.formula,
                                       weekend_df,
                                       return_type='dataframe')

        cov = self.model_res_weekend.cov_params()
        weekend_var = self.model_res_weekend.mse_resid + (weekend_X *
                                                          np.dot(cov, weekend_X.T).T).sum(1)
        weekend_var = pd.Series(weekend_var, index=weekend_df.index)
        weekday_var = pd.Series(weekday_var, index=weekday_df.index)

        variance_df = pd.concat([weekday_var, weekend_var])
        variance_df.sort_index(inplace=True)
        return variance_df

    def predict(self, df, summed=True):
        """
        Takes as input dataframe indexed with hour as frequency
        and with column tempF with hourly temparatures.
        Returns:
            if Summed is True, then returns summed prediction and variance
            as tuples
            Else, tuple of two series : prediction and varianes.
        """
        test_df = self.add_time_day(df)
        test_df = self.add_hdd(test_df)
        test_df = self.add_cdd(test_df)

        # We energy is what we predict, we don't require this column
        # except in compute_variance function which deconstructs dataframe
        # using patsy and use it to compute variancee. If this column
        # is not there then we just construct dummy here so that
        # compute_variance succeed.
        if 'energy' not in test_df:
            test_df = test_df.assign(energy=[0.0 for xx in test_df['tempF']])

        weekday_df = test_df.loc[test_df['day_of_week'].isin(self.weekdays)]
        weekday_pred = self.model_res_weekday.predict(weekday_df)

        weekend_df = test_df.loc[test_df['day_of_week'].isin(self.weekends)]
        weekend_pred = self.model_res_weekend.predict(weekend_df)

        # A series DS
        prediction = pd.concat([weekday_pred, weekend_pred])
        prediction.sort_index(inplace=True)

        # A Series DS
        variance = self.compute_variance(test_df)
        if summed:
            prediction = np.sum(prediction)
            variance = np.sum(variance)
        return prediction, variance
