import logging

from eemeter.weather.location import (
    zipcode_to_usaf_station,
    zipcode_to_tmy3_station,
    zipcode_to_cz2010_station,
)
from eemeter.co2.location import zipcode_to_avert_region
from eemeter.weather.eeweather_wrapper import WeatherSource
from eemeter.co2.avert import AVERTSource

logger = logging.getLogger(__name__)


def get_weather_source(site, use_cz2010=False):
    ''' Finds most relevant WeatherSource given project site.

    Parameters
    ----------
    site : eemeter.structures.ZIPCodeSite
        Site to match to weather source data.
    use_cz2010 : boolean, default False
        Indicates whether or not to use CZ2010 mapping.

    Returns
    -------
    weather_source : eemeter.weather.ISDWeatherSource or None
        Closest data-validated weather source in the same climate zone as
        project ZIP code, if available. If use_cz2010 is set, returns
        the ISDWeatherSource corresponding with the cz2010 station mapping.
        If no station can be found, returns None.
    '''

    zipcode = site.zipcode
    if use_cz2010:
        station = zipcode_to_cz2010_station(zipcode)
    else:
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
        weather_source = WeatherSource(station, normalized=False, use_cz2010=False)
    except ValueError:
        logger.error(
            "Could not create WeatherSource for station {}."
            .format(station)
        )
        return None

    logger.debug("Created ISDWeatherSource using station {}".format(station))

    return weather_source


def get_weather_normal_source(site, use_cz2010=False):
    ''' Finds most relevant WeatherSource given project site.

    Parameters
    ----------
    site : eemeter.structures.ZIPCodeSite
        Site to match to weather source data.
    use_cz2010 : boolean, default False
        Indicates whether or not to use CZ2010 mapping.

    Returns
    -------
    weather_normal_source : eemeter.weather.TMY3WeatherSource or eemeter.weather.CZ2010WeatherSource or None
        Closest data-validated TMY3 weather normal source in the same climate zone as
        project ZIP code, if available. If use_cz2010 is True, returns the
        corresponding CZ2010WeatherSource.
        If no station can be found, returns None.
    '''

    zipcode = site.zipcode

    if use_cz2010:
        station = zipcode_to_cz2010_station(zipcode)
        station_type = 'CZ2010'
    else:
        station = zipcode_to_tmy3_station(zipcode)
        station_type = 'TMY3'

    if station is None:
        logger.error(
            'Could not find appropriate {} weather normal station'
            ' for zipcode {}.'
            .format(station_type, zipcode)
        )
        return None

    logger.debug(
        'Mapped ZIP code {} to {} station {}'
        .format(zipcode, station_type, station)
    )

    try:
        weather_normal_source = WeatherSource(station, normalized=True, use_cz2010=use_cz2010)
    except ValueError:
        logger.error(
            "Could not create normalized WeatherSource for station {}."
            .format(station)
        )
        return None

    logger.debug(
        'Created {}WeatherSource using station {}'
        .format(station_type, station)
    )

    return weather_normal_source


def get_co2_source(site, use_year=2016):
    zipcode = site.zipcode
    region = zipcode_to_avert_region(zipcode)

    if region is None:
        logger.error(
            "Could not find AVERT region for zipcode {}."
            .format(zipcode)
        )
        return None

    logger.debug(
        "Mapped ZIP code {} to AVERT region {}"
        .format(zipcode, region)
    )

    try:
        avert_source = AVERTSource(use_year, region)
    except ValueError:
        logger.error(
            "Could not create AVERTSource for region {}."
            .format(region)
        )
        return None

    logger.debug("Created AVERTSource using region {}".format(region))

    return avert_source
