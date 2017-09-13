from collections import defaultdict
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error

from scipy.stats import linregress
import statsmodels.api as sm
import numpy as np
import patsy
import statsmodels.formula.api as smf

WEEK_END = 'WEEK_END'
WEEK_DAY = 'WEEK_DAY'

class DayOfWeekBasedLinearRegression(object):
    """
    A simple regression model based on "Day of Week", Temparature and hour
    of day features.
    Two separate linear regression models are created for--weeekdays and
    weekends.
    The fit function takes as input a dataframe indexed with hourly timestamps
    and tempF as column which contain hourly temparatures.
    """
    def __init__(self):
        self.model_weekday = None
        self.model_res_weekday = None
        self.model_weekend = None
        self.model_res_weekend = None
        self.weekdays = ['0', '1', '2', '3', '4']
        self.weekends = ['5', '6']

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

        new_df = df.assign(hour_of_day = hour_of_day,
                           day_of_week=day_of_week)

        return new_df

    def print_model_stats(self,
                          model_res):
        if not model_res:
            return
        rmse = np.sqrt(model_res.ssr/model_res.nobs)
        print model_res.params['Intercept'],  model_res.rsquared_adj,  "RMSE :", rmse

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
        weekday_df = train_df.loc[train_df['day_of_week'].isin(self.weekdays)]
        weekend_df = train_df.loc[train_df['day_of_week'].isin(self.weekends)]

        formulae = 'energy ~ tempF + hour_of_day + day_of_week + hour_of_day:day_of_week'

        try:
            self.model_weekday = smf.ols(formula=formulae, data=weekday_df)
            self.model_res_weekday = self.model_weekday.fit()
        except ValueError:
            self.model_weekday = None
            self.model_res_weekday = None

        try:
            self.model_weekend = smf.ols(formula=formulae, data=weekend_df)
            self.model_res_weekend = self.model_weekend.fit()
        except ValueError:
            self.model_weekend = None
            self.model_res_weekend = None

    def predict(self, df):
        """
        Takes as input dataframe indexed with hour as frequency
        and with column tempF with hourly temparatures.
        """
        test_df = self.add_time_day(df)
        prediction = []
        for row in test_df.itertuples():
            df = pd.DataFrame({
                'hour_of_day': [row.hour_of_day],
                'day_of_week' : [row.day_of_week],
                'tempF' : [row.tempF]
            })

            if row.day_of_week in self.weekdays:
                if self.model_res_weekday:
                    prediction.append(self.model_res_weekday.predict(df).get_value(0,0))
                else:
                    prediction.append(0.0)
            else:
                if self.model_res_weekend:
                    prediction.append(self.model_res_weekend.predict(df).get_value(0,0))
                else:
                    prediction.append(0.0)
        return pd.DataFrame({"energy_forecast" : prediction}, index=test_df.index)

