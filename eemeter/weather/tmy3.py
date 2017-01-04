from datetime import datetime, date

import pandas as pd
import pytz

from .base import WeatherSourceBase
from .clients import TMY3Client
from .cache import SqlJSONStore


class TMY3WeatherSource(WeatherSourceBase):
    ''' The :code:`TMY3WeatherSource` draws weather data from the NREL's
    Typical Meteorological Year 3 database. It stores fetched data locally by
    default in a SQLite database at :code:`~/.eemeter/cache/weather_cache.db`,
    unless you use set the EEMETER_WEATHER_CACHE_URL environment variable to
    another, SQLAlchemy compatible database URL:

    Basic usage is as follows:

    .. code-block:: python

        >>> from eemeter.weather import TMY3WeatherSource
        >>> ws = TMY3WeatherSource("724830")  # or another 6-digit USAF station

    This object can be used to fetch weather data as follows, using an daily
    frequency time-zone aware pandas DatetimeIndex covering any stretch
    of time.

    .. code-block:: python

        >>> import pandas as pd
        >>> import pytz
        >>> daily_index = pd.date_range('2015-01-01', periods=365,
        ...     freq='D', tz=pytz.UTC)
        >>> ws.indexed_temperatures(daily_index, "degF")
        2015-01-01 00:00:00+00:00    38.6450
        2015-01-02 00:00:00+00:00    40.4900
        2015-01-03 00:00:00+00:00    43.9175
                                      ...
        2015-12-29 00:00:00+00:00    43.7750
        2015-12-30 00:00:00+00:00    43.6250
        2015-12-31 00:00:00+00:00    46.9250
        Freq: D, dtype: float64
        >>> hourly_index = pd.date_range('2015-01-01', periods=365*24,
        ...     freq='H', tz=pytz.UTC)
        >>> ws.indexed_temperatures(hourly_index, "degF")
        2015-01-01 00:00:00+00:00    51.80
        2015-01-01 01:00:00+00:00    50.00
        2015-01-01 02:00:00+00:00    50.00
                                     ...
        2015-12-31 21:00:00+00:00    53.60
        2015-12-31 22:00:00+00:00    55.40
        2015-12-31 23:00:00+00:00    55.40
        Freq: H, dtype: float64

    '''

    cache_date_format = "%Y%m%d%H"
    cache_key_format = "TMY3-{}.json"
    freq = "H"
    client = TMY3Client()

    def __init__(self, station, cache_url=None, preload=True):
        super(TMY3WeatherSource, self).__init__(station)

        self.station = station
        self.json_store = SqlJSONStore(cache_url)

        self._check_station(station)

        if preload:
            self._load_data()

    def __repr__(self):
        return 'TMY3WeatherSource("{}")'.format(self.station)

    def _check_station(self, station):
        index = self.client._load_station_index()
        if station not in index:
            message = (
                "`{}` not recognized as valid TMY3 weather station identifier."
                .format(station)
            )
            raise ValueError(message)

    def _load_data(self):
        if self.json_store.key_exists(self._get_cache_key()):
            self.tempC = self._load_cached_series()
        else:
            self.tempC = self.client.get_tmy3_data(self.station)
            self._save_series(self.tempC)

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
