import logging

from eemeter.weather.location import (
    zipcode_to_usaf_station,
    zipcode_to_tmy3_station,
)
from eemeter.weather.noaa import ISDWeatherSource
from eemeter.weather.tmy3 import TMY3WeatherSource

logger = logging.getLogger(__name__)


def get_weather_source(site):
    ''' Finds most relevant WeatherSource given project site.

    Parameters
    ----------
    site : eemeter.structures.ZIPCodeSite
        Site to match to weather source data.

    Returns
    -------
    weather_source : eemeter.weather.ISDWeatherSource
        Closest data-validated weather source in the same climate zone as
        project ZIP code, if available.
    '''

    zipcode = site.zipcode
    station = zipcode_to_usaf_station(zipcode)

    if station is None:
        logger.error(
            "Could not find ISD station for zipcode {}."
            .format(zipcode)
        )
        return None

    logger.debug(
        "Mapped ZIP code {} to ISD station {}"
        .format(zipcode, station)
    )

    try:
        weather_source = ISDWeatherSource(station)
    except ValueError:
        logger.error(
            "Could not create ISDWeatherSource for station {}."
            .format(station)
        )
        return None

    logger.debug("Created ISDWeatherSource using station {}".format(station))

    return weather_source


def get_weather_normal_source(site):
    ''' Finds most relevant WeatherSource given project site.

    Parameters
    ----------
    site : eemeter.structures.ZIPCodeSite
        Site to match to weather source data.

    Returns
    -------
    weather_source : eemeter.weather.TMY3WeatherSource
        Closest data-validated weather source in the same climate zone as
        project ZIP code, if available.
    '''

    zipcode = site.zipcode
    station = zipcode_to_tmy3_station(zipcode)

    if station is None:
        logger.error(
            "Could not find appropriate TMY3 station for zipcode {}."
            .format(zipcode)
        )
        return None

    logger.debug(
        "Mapped ZIP code {} to TMY3 station {}"
        .format(zipcode, station)
    )

    try:
        weather_normal_source = TMY3WeatherSource(station)
    except ValueError:
        logger.error(
            "Could not create TMY3WeatherSource for station {}."
            .format(station)
        )
        return None

    logger.debug("Created TMY3WeatherSource using station {}".format(station))

    return weather_normal_source
