import pandas as pd
import pytz
import numpy as np


class MockWeatherClient(object):

    def _fake_temps(self, n):
        # sinusoidal fake temperatures in degC
        avg_temp = 15
        temp_range = 15
        period_offset = - (2 * np.pi / 3)
        temp_offsets = np.sin((2 * np.pi * np.arange(n) / n) + period_offset)
        return avg_temp + (temp_range * temp_offsets)

    def get_gsod_data(self, station, year):
        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 00:00".format(year),
                              freq='D', tz=pytz.UTC)
        temps = self._fake_temps(dates.shape[0])
        return pd.Series(temps, index=dates, dtype=float)

    def get_isd_data(self, station, year):
        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 23:00".format(year),
                              freq='H', tz=pytz.UTC)
        temps = self._fake_temps(dates.shape[0])
        return pd.Series(temps, index=dates, dtype=float)

    def get_hourly_weather_normal_data(self, station):
        dates = pd.date_range("1900-01-01 00:00",
                              "1900-12-31 23:00",
                              freq='H', tz=pytz.UTC)
        temps = self._fake_temps(dates.shape[0])
        return pd.Series(temps, index=dates, dtype=float)


class MockAVERTClient(object):

    def _fake_loads(self, n):
        # sinusoidal fake temperatures in degC
        avg_load = 25000.
        load_range = 10000.
        period_offset = - (2 * np.pi / 3)
        load_offsets = np.sin((2 * np.pi * np.arange(n) / n) + period_offset)
        return avg_load + (load_range * load_offsets)

    def read_rdf_file(self, year, region):
        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 23:00".format(year),
                              freq='D', tz=pytz.UTC)
        loads = self._fake_loads(dates.shape[0])
        co2_by_load = pd.Series(np.arange(0, 60000, 1000.),
                                np.arange(0, 60000, 1000.))
        return co2_by_load, pd.Series(loads, index=dates, dtype=float)
