from datetime import datetime, timedelta
import logging

import pandas as pd

from .base import WeatherSourceBase
from .clients import NOAAClient
from .cache import SqlJSONStore

logger = logging.getLogger(__name__)


class NOAAWeatherSourceBase(WeatherSourceBase):

    client = NOAAClient()

    def __init__(self, station, cache_url=None):
        super(NOAAWeatherSourceBase, self).__init__(station)

        self.json_store = SqlJSONStore(cache_url)
        self.loaded_years = set()
        self._check_station(station)
        logger.debug(
            "Created {} using cache: {}"
            .format(self, self.json_store)
        )
        self._check_for_recent_data()

    def _check_station(self, station):
        index = self.client._load_station_index()
        if station not in index:
            message = (
                "`{}` not recognized as valid USAF weather station identifier."
            )
            raise ValueError(message)

    def _check_for_recent_data(self, days_ago=1):
        target = datetime.now() - timedelta(days=days_ago)
        most_recent_fetch = self.json_store.retrieve_datetime(
            self._get_cache_key(target.year))
        if most_recent_fetch is not None:

            if target > most_recent_fetch:
                logger.debug(
                    "{} will update {} data because the most recent fetch"
                    " of that data occurred on {}, but the target date for"
                    " getting recent data is {}."
                    .format(self, target.year,
                            most_recent_fetch.strftime("%Y-%m-%d"),
                            target.strftime("%Y-%m-%d"))
                )
                self.add_year(target.year, force_fetch=True)
            else:
                logger.debug(
                    "{} will not update {} data because the most recent"
                    " fetch of that data occurred on {}, which is more recent"
                    " than the target date {}."
                    .format(self, target.year,
                            most_recent_fetch.strftime("%Y-%m-%d"),
                            target.strftime("%Y-%m-%d"))
                )
        else:
            logger.debug(
                "{self} will not update {year} data because {year} data is"
                " not cached."
                .format(self=self, year=target.year)
            )

    def add_year_range(self, start_year, end_year, force_fetch=False):
        """Adds temperature data to internal pandas timeseries across a
        range of years.

        .. note::

            This method is called automatically internally to keep data
            updated in response to calls to `.indexed_temperatures()`

        Parameters
        ----------
        start_year : {int, string}
            The earliest year for which data should be fetched, e.g. "2010".
        end_year : {int, string}
            The latest year for which data should be fetched, e.g. "2013".
        force_fetch : bool, default=False
            If True, forces the fetch; if false, checks to see if year
            has been added before actually fetching.
        """
        for year in range(start_year, end_year + 1):
            self.add_year(year, force_fetch)

    def add_year(self, year, force_fetch=False):
        """Adds temperature data to internal pandas timeseries

        .. note::

            This method is called automatically internally to keep data
            updated in response to calls to `.indexed_temperatures()`

        Parameters
        ----------
        year : {int, string}
            The year for which data should be fetched, e.g. "2010".
        force_fetch : bool, default=False
            If :code:`True`, forces the fetch; if :code:`False`, checks to see
            if locally available before actually fetching.
        """
        is_loaded = year in self.loaded_years
        self.loaded_years.add(year)
        if is_loaded:
            if force_fetch:  # it's loaded, but fetch anyway
                new_series = self._fetch_year(year)
                self.save_series(year, new_series)
                self.tempC.update(new_series)
                logger.debug(
                    "{} forced refetch of loaded {} data."
                    .format(self, year)
                )
            else:  # ignore request to add year since it's already added.
                logger.debug(
                    "{} ignored request to load {} data because it had"
                    " already been loaded."
                    .format(self, year)
                )
        else:
            if self._year_saved(year) and not force_fetch:
                # saved locally, no need to fetch
                new_series = self.load_series(year)
                logger.debug(
                    "{} loaded cached {} data."
                    .format(self, year)
                )
            else:  # not saved locally, need to fetch
                new_series = self._fetch_year(year)
                self.save_series(year, new_series)
                if force_fetch:
                    logger.debug(
                        "{} forced refetch of cached {} data."
                        .format(self, year)
                    )
                else:
                    logger.debug(
                        "{} performed initial fetch/cache of {} data."
                        .format(self, year)
                    )

            self.tempC = self._merge_series(self.tempC, new_series)

    def _get_cache_key(self, year):
        return self.cache_key_format.format(self.station, year)

    def _fetch_year(self, year):
        # get year from remote source
        message = "The `_fetch_year()` method must be implemented."
        raise NotImplementedError(message)

    def _year_saved(self, year):
        return self.json_store.key_exists(self._get_cache_key(year))

    def indexed_temperatures(self, index, unit, allow_mixed_frequency=False):
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

        if index.shape == (0,):
            return pd.Series([], index=index, dtype=float)

        self._verify_index_presence(index)  # fetches weather data if needed

        if index.freq is not None:
            freq = index.freq
        else:
            try:
                freq = pd.infer_freq(index)
            except ValueError:
                freq = None

        if freq == 'D':
            return self._daily_indexed_temperatures(index, unit)
        elif freq == 'H':
            return self._hourly_indexed_temperatures(index, unit)
        elif allow_mixed_frequency:
            return self._mixed_frequency_indexed_temperatures(index, unit)
        else:
            message = 'DatetimeIndex with unknown frequency not supported.'
            raise ValueError(message)

    def _daily_indexed_temperatures(self, index, unit):
        tempC = self.tempC.resample('D').mean()[index]
        return self._unit_convert(tempC, unit)

    def _hourly_indexed_temperatures(self, index, unit):
        message = (
            'DatetimeIndex frequency "H" not supported,'
            ' please resample to at least daily frequency ("D").'
            .format(index.freq)
        )
        raise ValueError(message)

    def _mixed_frequency_indexed_temperatures(self, index, unit):
        min_period = self._get_min_period(index)
        min_acceptable_period = self._get_min_acceptable_period()

        if min_period < min_acceptable_period:
            message = (
                'DatetimeIndex with a period below "{}" (found: {}) not'
                ' supported.'
                .format(min_acceptable_period, min_period)
            )
            raise ValueError(message)

        index_ = self._partitioned_multiindex(self.tempC.index, index)

        if index_ is None:
            message = 'Could not create partitioned mulitindex.'
            raise ValueError(message)

        level = index_.names[1]
        index_.get_level_values(level)
        values = self.tempC.reindex(index_.get_level_values(level)).values
        tempC = pd.DataFrame(values, index=index_)
        return self._unit_convert(tempC, unit)

    def _partitioned_multiindex(self, index_parts, index_periods, names=None):
        if names is None:
            if index_parts.freq == 'H':
                names = ["period", "hourly"]
            elif index_parts.freq == 'D':
                names = ["period", "daily"]
            else:
                message = (
                    'Unexpected temperature frequency "{}".'
                    .format(index_parts.freq)
                )
                raise ValueError(message)

        parts = iter(index_parts)
        periods = iter(index_periods)

        def _yield_index_tuples():
            part = next(parts, None)
            period_start = next(periods, None)
            period_end = next(periods, None)
            while part is not None:
                if period_start is None or period_end is None:
                    break
                if part < period_start:
                    part = next(parts, None)
                elif period_start <= part < period_end:
                    yield (period_start, part)
                    part = next(parts, None)
                else:
                    period_start = period_end
                    period_end = next(periods, None)

        index_tuples = list(_yield_index_tuples())
        if index_tuples == []:
            return None
        return pd.MultiIndex.from_tuples(index_tuples, names=names)

    def _get_min_period(self, index):
        return index.to_series().diff().dropna().min()

    def _get_min_acceptable_period(self):
        return pd.Timedelta('1 days')

    def _verify_index_presence(self, index):
        years = index.groupby(index.year).keys()
        for year in sorted(years):  # sorted for logging aesthetics
            self.add_year(year)

    def save_series(self, year, series):
        key = self._get_cache_key(year)
        data = [
            [
                d.strftime(self.cache_date_format), t
                if pd.notnull(t) else None
            ]
            for d, t in series.iteritems()
        ]
        self.json_store.save_json(key, data)

    def load_series(self, year):
        key = self._get_cache_key(year)
        data = self.json_store.retrieve_json(key)
        if data is None:
            raise KeyError("Key `{}` not found in cache.".format(key))

        index = pd.to_datetime([d[0] for d in data],
                               format=self.cache_date_format, utc=True)
        values = [d[1] for d in data]

        # changed for pandas > 0.18
        return pd.Series(values, index=index, dtype=float) \
            .sort_index().resample(self.freq).mean()

    def _merge_series(self, a, b):
        return a.append(b).sort_index().resample(self.freq).mean()


class GSODWeatherSource(NOAAWeatherSourceBase):
    ''' The :code:`GSODWeatherSource` draws weather data from the NOAA
    Global Summary of the Day FTP site. It stores fetched data locally by
    default in a SQLite database at :code:`~/eemeter/cache/weather_cache.db`,
    unless you use set the EEMETER_WEATHER_CACHE_URL environment variable to
    another, SQLAlchemy compatible database URL:

    Basic usage is as follows:

    .. code-block:: python

        >>> from eemeter.weather import GSODWeatherSource
        >>> ws = GSODWeatherSource("722880")  # or another 6-digit USAF station

    This object can be used to fetch weather data as follows, using an daily
    frequency time-zone aware pandas DatetimeIndex covering any stretch
    of time.

    .. code-block:: python

        >>> import pandas as pd
        >>> import pytz
        >>> index = pd.date_range('2015-01-01', periods=365,
        ...     freq='D', tz=pytz.UTC)
        >>> ws.indexed_temperatures(index, "degF")
        2015-01-01 00:00:00+00:00    43.6
        2015-01-02 00:00:00+00:00    45.0
        2015-01-03 00:00:00+00:00    47.3
                                     ...
        2015-12-29 00:00:00+00:00    48.0
        2015-12-30 00:00:00+00:00    46.4
        2015-12-31 00:00:00+00:00    47.6
        Freq: D, dtype: float64

    '''

    cache_date_format = "%Y%m%d"
    cache_key_format = "GSOD-{}-{}.json"
    year_existence_format = "{}-01-01"
    freq = "D"

    def __repr__(self):
        return 'GSODWeatherSource("{}")'.format(self.station)

    def _fetch_year(self, year):
        return self.client.get_gsod_data(self.station, year)


class ISDWeatherSource(NOAAWeatherSourceBase):
    ''' The :code:`ISDWeatherSource` draws weather data from the NOAA
    Integrated Surface Database (ISD) FTP site. It stores fetched hourly data
    locally by default in a SQLite database at
    :code:`~/eemeter/cache/weather_cache.db`, unless you use set the following
    environment variable to something different:

    .. code-block:: bash

        $ export EEMETER_WEATHER_CACHE_DIRECTORY=/path/to/custom/directory

    Basic usage is as follows:

    .. code-block:: python

        >>> from eemeter.weather import ISDWeatherSource
        >>> ws = ISDWeatherSource("722880")  # or another 6-digit USAF station

    This object can be used to fetch weather data as follows, using an hourly
    or daily frequency time-zone aware pandas DatetimeIndex covering any
    stretch of time.

    .. code-block:: python

        >>> import pandas as pd
        >>> import pytz
        >>> daily_index = pd.date_range('2015-01-01', periods=365,
        ...     freq='D', tz=pytz.UTC)
        >>> ws.indexed_temperatures(daily_index, "degF")
        2015-01-01 00:00:00+00:00    43.550000
        2015-01-02 00:00:00+00:00    45.042500
        2015-01-03 00:00:00+00:00    47.307500
                                       ...
        2015-12-29 00:00:00+00:00    47.982500
        2015-12-30 00:00:00+00:00    46.415000
        2015-12-31 00:00:00+00:00    47.645000
        Freq: D, dtype: float64
        >>> hourly_index = pd.date_range('2015-01-01', periods=365*24,
        ...     freq='H', tz=pytz.UTC)
        >>> ws.indexed_temperatures(hourly_index, "degF")
        2015-01-01 00:00:00+00:00    51.98
        2015-01-01 01:00:00+00:00    50.00
        2015-01-01 02:00:00+00:00    48.02
                                     ...
        2015-12-31 21:00:00+00:00    62.06
        2015-12-31 22:00:00+00:00    62.06
        2015-12-31 23:00:00+00:00    62.06
        Freq: H, dtype: float64

    '''

    cache_date_format = "%Y%m%d%H"
    cache_key_format = "ISD-{}-{}.json"
    year_existence_format = "{}-01-01 00"
    freq = "H"

    def __repr__(self):
        return 'ISDWeatherSource("{}")'.format(self.station)

    def _fetch_year(self, year):
        return self.client.get_isd_data(self.station, year)

    def _hourly_indexed_temperatures(self, index, unit):
        tempC = self.tempC.resample(self.freq).mean()[index]
        return self._unit_convert(tempC, unit)

    def _get_min_acceptable_period(self):
        return pd.Timedelta('1 hours')
