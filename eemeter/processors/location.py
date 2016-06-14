from eemeter.weather.location import zipcode_to_station
from eemeter.weather.noaa import ISDWeatherSource
from eemeter.weather.tmy import TMY3WeatherSource


class WeatherSourceProcessor(object):

    def get_weather_source(self, location):

        station = zipcode_to_station(location.zipcode)
        weather_source = ISDWeatherSource(station)
        validation_errors = []

        return weather_source, validation_errors

    def get_weather_normal_source(self, location):

        station = zipcode_to_station(location.zipcode)
        weather_source = TMY3WeatherSource(station)
        validation_errors = []

        return weather_source, validation_errors
