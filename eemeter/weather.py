import ftplib
import StringIO
import gzip
from datetime import datetime
from datetime import timedelta
import numpy as np
import requests

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
        self._data = {}
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login()
        data = []
        for year in xrange(start_year,end_year + 1):
            string = StringIO.StringIO()
            ftp.retrbinary('RETR /pub/data/gsod/{year}/{station_id}-{year}.op.gz'.format(station_id=station_id,year=year),string.write)
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
    result = requests.get('http://developer.nrel.gov/api/solar/data_query/v1.json?api_key={}&lat={}&lon={}'.format(api_key,lat,lng))
    return result.json()['outputs']['tmy3']['id'][2:]

def ziplocate_us(zipcode):
    result = requests.get('http://ziplocate.us/api/v1/{}'.format(zipcode))
    data = result.json()
    return data.get('lat'), data.get('lng')
