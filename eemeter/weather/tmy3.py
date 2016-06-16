from .cache import CachedWeatherSourceBase
from .clients import TMY3Client

from datetime import datetime, timedelta, date, time

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
            start_date = datetime(1900, 1, 1, tzinfo=pytz.UTC)
            index = [
                start_date + timedelta(seconds=i*3600)
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
        self.tempC = pd.Series(temps, index=index, dtype=float) \
            .sort_index().resample('H').mean()
        self.save_to_cache()

    @staticmethod
    def _normalize_datetime(dt, year_offset=0):
        return datetime(1900 + year_offset, dt.month, dt.day, dt.hour,
                        dt.minute, dt.second, tzinfo=pytz.UTC)

    @staticmethod
    def _get_loffset(timestamp):
        t = timestamp.time()
        return datetime.combine(date(1, 1, 1), t) - datetime(1, 1, 1, 0, 0, 0)

    def _normalize_index(self, index):
        return pd.DatetimeIndex([self._normalize_datetime(dt) for dt in index])

    def indexed_temperatures(self, index, unit):
        ''' Return average temperatures over the given index.

        Parameters
        ----------
        index : pandas.DatetimeIndex
            Index over which to supply average temperatures.
            The :code:`index` should be given as either an hourly ('H') or
            daily ('D') frequency.
        unit : str, {"degF", "degC"}
            Target temperature unit for returned temperature series.

        Returns
        -------
        temperatures : pandas.Series with DatetimeIndex
            Average temperatures over series indexed by :code:`index`.
        '''

        if index.freq == 'D':
            return self._daily_indexed_temperatures(index, unit)
        elif index.freq == 'H':
            return self._hourly_indexed_temperatures(index, unit)
        else:
            message = (
                'DatetimeIndex frequency "{}" not supported, please resample.'
                .format(index.freq)
            )
            raise ValueError(message)

    def _daily_indexed_temperatures(self, index, unit):
        normalized_index = self._normalize_index(index)
        loffset = self._get_loffset(normalized_index[0])
        tempC = self.tempC.resample('D', loffset=loffset) \
            .mean()[normalized_index]
        tempC.index = index
        return self._unit_convert(tempC, unit)

    def _hourly_indexed_temperatures(self, index, unit):
        normalized_index = self._normalize_index(index)
        tempC = self.tempC.resample('H').mean()[normalized_index]
        tempC.index = index
        return self._unit_convert(tempC, unit)
