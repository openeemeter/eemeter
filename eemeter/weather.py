import ftplib
import StringIO
import gzip
import os
import json
from datetime import datetime
from datetime import timedelta
import numpy as np
import requests
import eemeter

class WeatherSourceBase:
    def get_average_temperature(self,consumption_history,fuel_type):
        """Returns a list of floats containing the average temperature during
        each consumption period.
        """
        avg_temps = []
        # TODO - WARNING - deal with the fact that consumption history.get will
        # not return things in a predictable order.
        for consumption in consumption_history.get(fuel_type):
            avg_temps.append(self.get_consumption_average_temperature(consumption))
        return avg_temps

    def get_consumption_average_temperature(self,consumption):
        raise NotImplementedError

class GSODWeatherSource(WeatherSourceBase):
    def __init__(self,station_id,start_year,end_year):
        if len(station_id) == 6:
            # given station id is the six digit code, so need to get full name
            gsod_station_index_filename = os.path.join(
                    os.path.dirname(os.path.dirname(eemeter.__file__)),
                    'resources',
                    'GSOD-ISD_station_index.json')
            with open(gsod_station_index_filename,'r') as f:
                station_index = json.load(f)
            # take first station in list
            potential_station_ids = station_index[station_id]
        else:
            # otherwise, just use the given id
            potential_station_ids = [station_id]
        self._data = {}
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login()
        data = []
        for year in xrange(start_year,end_year + 1):
            string = StringIO.StringIO()
            # not every station will be available in every year, so use the
            # first one that works
            for station_id in potential_station_ids:
                try:
                    ftp.retrbinary('RETR /pub/data/gsod/{year}/{station_id}-{year}.op.gz'.format(station_id=station_id,year=year),string.write)
                    break
                except (IOError,ftplib.error_perm):
                    pass
            string.seek(0)
            f = gzip.GzipFile(fileobj=string)
            self._add_file(f)
            string.close()
            f.close()
        ftp.quit()

    def get_consumption_average_temperature(self,consumption):
        """Gets the average temperature during a particular Consumption
        instance. Resolution limit: daily.
        """
        avg_temps = []
        for days in xrange(consumption.timedelta.days):
            avg_temps.append(self._data[(consumption.start + timedelta(days=days)).strftime("%Y%m%d")])
        return np.mean(avg_temps)

    def _add_file(self,f):
        for line in f.readlines()[1:]:
            columns=line.split()
            self._data[columns[2]] = float(columns[3])

class ISDWeatherSource(WeatherSourceBase):
    def __init__(self,station_id,start_year,end_year):
        if len(station_id) == 6:
            # given station id is the six digit code, so need to get full name
            gsod_station_index_filename = os.path.join(
                    os.path.dirname(os.path.dirname(eemeter.__file__)),
                    'resources',
                    'GSOD-ISD_station_index.json')
            with open(gsod_station_index_filename,'r') as f:
                station_index = json.load(f)
            # take first station in list
            potential_station_ids = station_index[station_id]
        else:
            # otherwise, just use the given id
            potential_station_ids = [station_id]
        self._data = {}
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login()
        data = []
        for year in xrange(start_year,end_year + 1):
            string = StringIO.StringIO()
            # not every station will be available in every year, so use the
            # first one that works
            for station_id in potential_station_ids:
                try:
                    ftp.retrbinary('RETR /pub/data/noaa/{year}/{station_id}-{year}.gz'.format(station_id=station_id,year=year),string.write)
                    break
                except (IOError,ftplib.error_perm):
                    pass
            string.seek(0)
            f = gzip.GzipFile(fileobj=string)
            self._add_file(f)
            string.close()
            f.close()
        ftp.quit()
        pass

    def get_consumption_average_temperature(self,consumption):
        """Gets the average temperature during a particular Consumption
        instance. Resolution limit: hourly.
        """
        avg_temps = []
        n_hours = consumption.timedelta.days * 24 + consumption.timedelta.seconds // 3600
        for hours in xrange(n_hours):
            hour = consumption.start + timedelta(seconds=hours*3600)
            hourly = self._data.get(hour.strftime("%Y%m%d%H"),float("nan"))
            avg_temps.append(hourly)
        # mask nans
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def _add_file(self,f):
        for line in f.readlines():
            # line[4:10] # USAF
            # line[10:15] # WBAN
            # line[28:34] # latitude
            # line[34:41] # longitude
            # line[46:51] # elevation
            # line[92:93] # temperature reading quality
            # year = line[15:19]
            # month = line[19:21]
            # day = line[21:23]
            # hour = line[23:25]
            # minute = line[25:27]
            air_temperature = (float(line[87:92]) / 10) * 1.8 + 32
            if line[87:92] == "+9999":
                air_temperature = float("nan")
            self._data[line[15:25]] = air_temperature

class WeatherUndergroundWeatherSource(WeatherSourceBase):
    def __init__(self,zipcode,start,end,api_key):
        self._data = {}
        assert end >= start
        date_format = "%Y%m%d"
        date_range_limit = 32
        end_date_str = datetime.strftime(end, date_format)
        for days in range(0, (end - start).days, date_range_limit):
            start_date = start + timedelta(days=days)
            start_date_str = datetime.strftime(start_date, date_format)
            query = 'http://api.wunderground.com/api/{}/history_{}{}/q/{}.json'\
                    .format(api_key,start_date_str,end_date_str,zipcode)
            self._get_query_data(query)

    def get_consumption_average_temperature(self,consumption):
        """Gets the average temperature during a particular Consumption
        instance. Resolution limit: daily.
        """
        avg_temps = []
        for days in xrange(consumption.timedelta.days):
            avg_temps.append(self._data[(consumption.start + timedelta(days=days)).strftime("%Y%m%d")]["meantempi"])
        return np.mean(avg_temps)

    def _get_query_data(self,query):
        for day in requests.get(query).json()["history"]["dailysummary"]:
            date_string = day["date"]["year"] + day["date"]["mon"] + day["date"]["mday"]
            data = {"meantempi":int(day["meantempi"])}
            self._data[date_string] = data

def nrel_tmy3_station_from_lat_long(lat,lng,api_key):
    """Use the National Renewable Energy Lab API to find the closest weather
    station for a particular lat/long. Requires a (freely available) API key.
    """
    result = requests.get('http://developer.nrel.gov/api/solar/data_query/v1.json?api_key={}&lat={}&lon={}'.format(api_key,lat,lng))
    return result.json()['outputs']['tmy3']['id'][2:]

def ziplocate_us(zipcode):
    """Use the Ziplocate API to find the population-weighted lat/long centroid
    for this ZIP code.
    """
    result = requests.get('http://ziplocate.us/api/v1/{}'.format(zipcode))
    if result.status_code == 200:
        data = result.json()
        return data.get('lat'), data.get('lng')
    elif result.status_code == 404:
        raise ValueError("No known lat/long centroid for this ZIP code.")
    else:
        return None

def usaf_station_from_zipcode(zipcode,nrel_api_key):
    lat,lng = ziplocate_us(zipcode)
    station = nrel_tmy3_station_from_lat_long(lat,lng,nrel_api_key)
    return station

