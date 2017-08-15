from .base import NormalHourlyWeatherSourceBase
from .clients import CZ2010Client


class CZ2010WeatherSource(NormalHourlyWeatherSourceBase):
    ''' In some cases in California, regulations require using CZ2010 weather
    normal data. This weather source uses CZ2010 data. It uses the same
    weather cache as TMY3 or ISD Weather sources, see those docs for more info.

    Basic usage is as follows:

    .. code-block:: python

        >>> from eemeter.weather import CZ2010WeatherSource
        >>> ws = CZ2010WeatherSource("724830")  # or another 6-digit USAF station

    Other usage is the same as for the TMY3WeatherSource

    '''

    station_type = 'CZ2010'
    client = CZ2010Client()
