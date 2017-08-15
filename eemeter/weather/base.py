from datetime import datetime, date

import pandas as pd
import pytz

from .cache import SqlJSONStore


class WeatherSourceBase(object):

    def __init__(self, station):
        self.station = station
        self.tempC = pd.Series(dtype=float)

    @staticmethod
    def _unit_convert(x, unit):
        if unit is None or unit == "degC":
            return x
        elif unit == "degF":
            return 1.8 * x + 32
        else:
            message = (
                "Unit not supported ({}). Use 'degF' or 'degC'"
                .format(unit)
            )
            raise ValueError(message)


class NormalHourlyWeatherSourceBase(WeatherSourceBase):
    '''Base class for hourly-frequency normal weather sources.

    Must define station_type and client class attributes. See TMY3WeatherSource
    or CZ2010 classes as examples.
    '''

    cache_date_format = "%Y%m%d%H"
    cache_key_format = "{}-{}.json"
    freq = "H"
    # station_type = '...'  # inheriting classes should define this
    # client = XXXClient()  # client must define client.get_hourly_weather_normal_data(station)

    def __init__(self, station, cache_url=None, preload=True):
        super(NormalHourlyWeatherSourceBase, self).__init__(station)

        self.station = station
        self.json_store = SqlJSONStore(cache_url)

        self._check_station(station)

        if preload:
            self._load_data()

    def __repr__(self):
        return '{}WeatherSource("{}")'.format(self.station_type, self.station)

    def _load_data(self):
        if self.json_store.key_exists(self._get_cache_key()):
            self.tempC = self._load_cached_series()
        else:
            self.tempC = self.client.get_hourly_weather_normal_data(self.station)
            self._save_series(self.tempC)

    def _check_station(self, station):
        index = self.client._load_station_index()
        if station not in index:
            message = (
                "`{}` not recognized as valid {} weather station identifier."
                .format(self.station_type, station)
            )
            raise ValueError(message)

    def _load_cached_series(self):
        data = self.json_store.retrieve_json(self._get_cache_key())

        index = pd.to_datetime([d[0] for d in data],
                               format=self.cache_date_format, utc=True)
        values = [d[1] for d in data]

        # changed for pandas > 0.18
        return pd.Series(values, index=index, dtype=float) \
            .sort_index().resample(self.freq).mean()

    def _save_series(self, series):
        data = [
            [
                d.strftime(self.cache_date_format), t
                if pd.notnull(t) else None
            ]
            for d, t in series.iteritems()
        ]
        self.json_store.save_json(self._get_cache_key(), data)

    def _get_cache_key(self):
        return self.cache_key_format.format(self.station_type, self.station)

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
