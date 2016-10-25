import pandas as pd
import pytz


class MockWeatherClient(object):

    def get_gsod_data(self, station, year):
        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 00:00".format(year),
                              freq='D', tz=pytz.UTC)
        return pd.Series(0, index=dates, dtype=float)

    def get_isd_data(self, station, year):
        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 23:00".format(year),
                              freq='H', tz=pytz.UTC)
        return pd.Series(0, index=dates, dtype=float)

    def get_tmy3_data(self, station):
        index = pd.date_range("1900-01-01 00:00",
                              "1900-12-31 23:00",
                              freq='H', tz=pytz.UTC)
        return pd.Series(0, index=index, dtype=float)


class MockModel(object):

    def __init__(self):
        self.n = 1
        self.upper = 1
        self.lower = 1

    def fit(self, df):
        return {}

    def predict(self, df, params=None, summed=True):
        if summed:
            return pd.Series(1, index=df.index).sum(), 1, 1
        else:
            return pd.Series(1, index=df.index), 1, 1
