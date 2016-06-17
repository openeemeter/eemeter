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

