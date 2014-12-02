import ftplib
import StringIO
import gzip
from datetime import datetime
from datetime import timedelta
import numpy as np

class WeatherGetterBase:
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

class GSODWeatherGetter(WeatherGetterBase):
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
