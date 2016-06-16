from .base import WeatherSourceBase
from .clients import TMY3Client
from .cache import SqliteJSONStore

from datetime import datetime, date

import pandas as pd
import pytz


class TMY3WeatherSource(WeatherSourceBase):

    cache_date_format = "%Y%m%d%H"
    cache_key_format = "TMY3-{}.json"
    freq = "H"
    client = TMY3Client()

    def __init__(self, station, cache_directory=None):
        super(TMY3WeatherSource, self).__init__(station)

        self.station = station
        self.json_store = SqliteJSONStore(cache_directory)

        self._load_data()

    def load_cached_series(self):
        data = self.json_store.retrieve_json(self.get_cache_key())

        index = pd.to_datetime([d[0] for d in data],
                               format=self.cache_date_format, utc=True)
        values = [d[1] for d in data]

        # changed for pandas > 0.18
        return pd.Series(values, index=index, dtype=float) \
            .sort_index().resample(self.freq).mean()

    def _load_data(self):
        if self.json_store.key_exists(self.get_cache_key()):
            self.tempC = self.load_cached_series()
        else:
            self.tempC = self.client.get_tmy3_data(self.station)
            self.save_series(self.tempC)

    def save_series(self, series):
        data = [
            [
                d.strftime(self.cache_date_format), t
                if pd.notnull(t) else None
            ]
            for d, t in series.iteritems()
        ]
        self.json_store.save_json(self.get_cache_key(), data)

    def get_cache_key(self):
        return self.cache_key_format.format(self.station)

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
