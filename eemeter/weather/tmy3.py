from eemeter.evaluation import Period
from .cache import CachedWeatherSourceBase
from .clients import TMY3Client

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

class TMY3WeatherSource(CachedWeatherSourceBase):

    cache_date_format = "%Y%m%d%H"
    cache_filename_format = "TMY3-{}.json"
    freq = "H"
    client = TMY3Client()

    def __init__(self, station, station_fallback=True):
        super(TMY3WeatherSource, self).__init__(station)
        self.station_id = station
        self.station_fallback = station_fallback
        if self.tempC.shape[0] != 365 * 24:
            self._load_data()

    def _load_data(self):
        data = self.client.get_tmy3_data(self.station, self.station_fallback)
        if data is None:
            temps = [np.nan for _ in range(365 * 24)]
            index = [
                datetime(1900, 1, 1, tzinfo=pytz.UTC)
                + timedelta(seconds=i*3600)
                for i in range(365 * 24)
            ]
        else:
            temps = [d["temp_C"] for d in data]
            index = [
                datetime(1900, d["dt"].month, d["dt"].day, d["dt"].hour,
                         tzinfo=pytz.UTC)
                for d in data
            ]

        # changed for pandas > 0.18
        self.tempC = pd.Series(temps, index=index, dtype=float).sort_index().resample('H').mean()
        self.save_to_cache()

    def annual_daily_temperatures(self, unit):
        """Returns a list of daily temperature normals for a typical
        meteorological year.

        Parameters
        ----------
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            List with single element which is an array of observed
            daily temperatures.

        """

        period = Period(start=datetime(1900,1,1, tzinfo=pytz.UTC),
                        end=datetime(1901,1,1, tzinfo=pytz.UTC))
        return self.daily_temperatures([period], unit)

    def _fetch_period(self, period):
        pass # loaded at init

    def _fetch_datetime(self, dt):
        pass # loaded at init

    def _normalize_period(self, period):
        start = self._normalize_datetime(period.start)
        year_offset = period.end.year - period.start.year
        end = self._normalize_datetime(period.end, year_offset)
        return Period(start, end)

    @staticmethod
    def _normalize_datetime(dt, year_offset=0):
        return datetime(1900 + year_offset, dt.month, dt.day, dt.hour, dt.minute, dt.second, tzinfo=pytz.UTC)

    def _period_average_temperature(self, period, unit):
        period = self._normalize_period(period)
        return super(TMY3WeatherSource, self)._period_average_temperature(period, unit)

    def _period_daily_temperatures(self, period, unit):
        period = self._normalize_period(period)
        return super(TMY3WeatherSource, self)._period_daily_temperatures(period, unit)

    def _period_hourly_temperatures(self, period, unit):
        period = self._normalize_period(period)
        return super(TMY3WeatherSource, self)._period_hourly_temperatures(period, unit)

    def datetime_average_temperature(self, dt, unit):
        """The daily average temperatures for a particular period.

        Parameters
        ----------
        dt : datetime.datetime or datetime.date
            The date for which to find or calculate the average temperature.
        unit : {"degC", "degF"}
            The unit in which the daily average temperature should be returned.

        Returns
        -------
        out : np.ndarray
            Average temperature observed.
        """
        dt = self._normalize_datetime(dt)
        return super(TMY3WeatherSource, self).datetime_average_temperature(dt, unit)

    def datetime_hourly_temperature(self, dt, unit):
        """The hourly observed temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
            Time periods over which temperatures will be collected. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of arrays of observed_daily temperatures observed during each
            period. Note: array is not guaranteed to be rectangular. If a
            single datetime period is given, a single numpy array of
            temperatures will be returned.
        """
        dt = self._normalize_datetime(dt)
        return super(TMY3WeatherSource, self).datetime_hourly_temperature(dt, unit)
