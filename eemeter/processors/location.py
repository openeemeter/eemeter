from eemeter.weather.location import zipcode_to_station
from eemeter.weather.noaa import ISDWeatherSource
from eemeter.weather.tmy3 import TMY3WeatherSource
from eemeter.processors.collector import collects


@collects()
def get_weather_source(project):

    zipcode = project.site.zipcode

    logs = {}

    try:
        station = zipcode_to_station(zipcode)
    except KeyError:
        message = "Could not find station for zipcode {}." .format(zipcode)
    logs["ISD_station_id"] = station

    try:
        weather_source = ISDWeatherSource(station)
    except ValueError:
        message = (
            "Could not create ISDWeatherSource for station {}."
            .format(station)
        )
    else:
        message = "Created ISDWeatherSource for station {}.".format(station)
    notes["ISD_station_to_weather_source"] = message

    return weather_source, notes


@collects()
def site_to_weather_normal_source(site):

    station = zipcode_to_station(site.zipcode)
    weather_source = TMY3WeatherSource(station)
    validation_errors = []

    return weather_source, validation_errors
