from .base import NormalHourlyWeatherSourceBase
from .clients import TMY3Client


class TMY3WeatherSource(NormalHourlyWeatherSourceBase):
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

    station_type = 'TMY3'
    client = TMY3Client()
