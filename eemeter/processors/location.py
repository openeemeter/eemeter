from eemeter.weather.location import zipcode_to_station
from eemeter.weather.noaa import ISDWeatherSource
from eemeter.weather.tmy3 import TMY3WeatherSource


def get_weather_source(logger, project):

    zipcode = project.site.zipcode

    try:
        station = zipcode_to_station(zipcode)
    except KeyError:
        logger.error(
            "Could not find ISD station for zipcode {}."
            .format(zipcode)
        )
        return None

    logger.info(
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

    logger.info("Created ISDWeatherSource using station {}".format(station))

    return weather_source


def get_weather_normal_source(logger, project):

    zipcode = project.site.zipcode

    logs = {}

    try:
        station = zipcode_to_station(zipcode)
    except KeyError:
        logger.error(
            "Could not find TMY3 station for zipcode {}."
            .format(zipcode)
        )
        return None

    logger.info(
        "Mapped ZIP code {} to TMY3 station {}"
        .format(zipcode, station)
    )

    try:
        weather_source = TMY3WeatherSource(station)
    except ValueError:
        logger.error(
            "Could not create TMY3WeatherSource for station {}."
            .format(station)
        )
        return None

    logger.info("Created ISDWeatherSource using station {}".format(station))

    return weather_normal_source
