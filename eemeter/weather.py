import ftplib
from io import BytesIO
import gzip
import os
import json
from datetime import datetime
from datetime import date
from datetime import timedelta
import warnings
import numpy as np
import requests
from . import ureg, Q_
from pkg_resources import resource_stream

Session = None

try:
    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.orm import relationship, backref
    from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
    from sqlalchemy.orm.exc import NoResultFound

    Base = declarative_base()

    class WeatherStation(Base):
        __tablename__ = 'weatherstation'

        id = Column(Integer, primary_key=True)
        usaf_id = Column(String)
        temperature_normals = relationship("TemperatureNormal", order_by="TemperatureNormal.date", backref="weatherstation")
        average_temperatures = relationship("AverageTemperature", order_by="AverageTemperature.date", backref="weatherstation")

        def __repr__(self):
               return "<WeatherStation('{}')".format(self.usaf_id)

    class AverageTemperature(Base):
        __tablename__ = 'averagetemperature'

        id = Column(Integer, primary_key=True)
        weatherstation_id = Column(Integer, ForeignKey('weatherstation.id'))
        temp_C = Column(Float)
        date = Column(Date)

        def __repr__(self):
               return "<AverageTemperature('{}', {}, {})>".format(
                                    self.weatherstation.usaf_id, self.temp_C, self.date)

    class TemperatureNormal(Base):
        __tablename__ = 'temperaturenormal'

        id = Column(Integer, primary_key=True)
        weatherstation_id = Column(Integer, ForeignKey('weatherstation.id'))
        temp_C = Column(Float)
        date = Column(Date)

        def __repr__(self):
               return "<TemperatureNormal('{}', {}, {})>".format(
                                    self.weatherstation.usaf_id, self.temp_C, self.date)

except ImportError:
    warnings.warn("cache disabled. To use, please install sqlalchemy.")

def initialize_cache():
    cache_db_url = os.environ.get("EEMETER_WEATHER_CACHE_DATABASE_URL")
    if cache_db_url is None:
        warnings.warn("cache disabled. To use, please set the EEMETER_WEATHER_CACHE_DATABASE_URL environment variable.")
        return None
    engine = create_engine(cache_db_url)
    Session = sessionmaker()
    Session.configure(bind=engine)
    Base.metadata.create_all(engine)
    return Session

class WeatherSourceBase(object):

    def __init__(self):


        global Session
        if not Session:
            try:
                Session = initialize_cache()
            except NameError:
                Session = None

        if Session:
            self.session = Session()
        else:
            self.session = None

        self.data = {}
        self.source_unit = ureg.degC
        self.station_id = None

    def get_average_temperature(self,periods,unit_name):
        """Returns a list of average temperatures of each DatetimePeriod in
        the given unit (usually "degF" or "degC").
        """
        unit = ureg.parse_expression(unit_name)
        avg_temps = []
        for period in periods:
            avg_temps.append(self.get_period_average_temperature(period,unit))
        return avg_temps

    def get_period_average_temperature(self,period,unit):
        """Returns the average temperature during the duration of a single
        DatetimePeriod instance as calculated by taking the mean of daily average
        temperatures.
        """
        avg_temps = self.get_period_daily_temperatures(period,unit)
        return np.mean(avg_temps)

    def get_daily_temperatures(self,periods,unit_name):
        """Returns, for each period, a list of average daily temperatures
        observed during the duration of that period. Return value
        is a list of lists of temperatures.
        """
        unit = ureg.parse_expression(unit_name)
        daily_temps = []
        for period in periods:
            daily_temps.append(self.get_period_daily_temperatures(period,unit))
        return daily_temps

    def get_period_daily_temperatures(self,period,unit):
        """Returns, for a particular period instance, a list of average
        daily temperatures observed during the duration of that period
        period. Result is a list of temperatures.
        """
        avg_temps = []
        for days in range(period.timedelta.days):
            day = period.start + timedelta(days=days)
            temp = self.get_daily_average_temperature(day,unit)
            avg_temps.append(temp)
        return np.array(avg_temps)

    def get_daily_average_temperature(self,day,unit):
        """Should return the average temperature of the given day. Must be
        implemented by inheriting classes.
        """
        raise NotImplementedError

    def get_hdd(self,periods,unit_name,base):
        """Returns, for each period, the total heating degree days
        observed during the period.
        """
        unit = ureg.parse_expression(unit_name)
        hdds = []
        for period in periods:
            hdds.append(self.get_period_hdd(period,unit,base))
        return hdds

    def get_period_hdd(self,period,unit,base):
        """Returns the total heating degree days observed during the
        period.
        """
        total_hdd = 0.
        for days in range(period.timedelta.days):
            day = period.start + timedelta(days=days)
            temp = self.get_daily_average_temperature(day,unit)
            if temp < base:
                total_hdd += base - temp
        return total_hdd

    def get_cdd(self,periods,unit_name,base):
        """Returns, for each period, the total cooling degree days
        observed during the period.
        """
        unit = ureg.parse_expression(unit_name)
        cdds = []
        for period in periods:
            cdds.append(self.get_period_cdd(period,unit,base))
        return cdds

    def get_period_cdd(self,period,unit,base):
        """Returns the total cooling degree days observed during the
        period.
        """
        total_cdd = 0.
        for days in range(period.timedelta.days):
            day = period.start + timedelta(days=days)
            temp = self.get_daily_average_temperature(day,unit)
            if temp > base:
                total_cdd += temp - base
        return total_cdd

class CachedAverageTemperatureDataMixin(object):

    def init_average_temperature_data(self):
        if self.session:
            try:
                self.weather_station = self.session.query(WeatherStation).filter(WeatherStation.usaf_id == self.station_id).one()
            except NoResultFound:
                self.weather_station = WeatherStation(usaf_id=self.station_id)
                self.session.add(self.weather_station)
                self.session.commit()
            for t in self.weather_station.average_temperatures:
                self.data[t.date.strftime("%Y%m%d")] = Q_(t.temp_C,ureg.degC)
        else:
            self.weather_station = None

    def update_cache(self,temp_C,date,overwrite=True):
        if self.session:
            temps_query = self.session.query(AverageTemperature)\
                        .filter(AverageTemperature.weatherstation == self.weather_station)\
                        .filter(AverageTemperature.date==date)

            temps = temps_query.all()
            if overwrite:
                for t in temps:
                    self.session.delete(t)
                self.session.commit()
                temps = []
            if temps == []:
                t = AverageTemperature(weatherstation=self.weather_station,temp_C=temp_C,date=date)
                self.session.add(t)
                self.session.commit()

class GSODWeatherSource(WeatherSourceBase,CachedAverageTemperatureDataMixin):
    def __init__(self,station_id,start_year,end_year):
        super(GSODWeatherSource,self).__init__()
        self.source_unit = ureg.degF
        self.station_id = station_id[:6]
        self.init_average_temperature_data()

        for days in range((datetime(end_year,12,31) - datetime(start_year,1,1)).days):
            dat = datetime(start_year,1,1) + timedelta(days=days)
            if dat > datetime.now() - timedelta(days=1):
                break
            temp = self.data.get(dat.strftime("%Y%m%d"))
            if temp is None:
                self._fetch_year(dat.year)

    def _fetch_year(self,year):
        if len(self.station_id) == 6:
            # given station id is the six digit code, so need to get full name
            with resource_stream('eemeter.resources','GSOD-ISD_station_index.json') as f:
                station_index = json.loads(f.read().decode("utf-8"))
            # take first station in list
            potential_station_ids = station_index[self.station_id]
        else:
            # otherwise, just use the given id
            potential_station_ids = [self.station_id]

        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login()

        string = BytesIO()

        # not every station will be available in every year, so use the
        # first one that works
        for station_id in potential_station_ids:
            try:
                ftp.retrbinary('RETR /pub/data/gsod/{year}/{station_id}-{year}.op.gz'.format(station_id=station_id,year=year),string.write)
                break
            except (IOError,ftplib.error_perm):
                pass
        ftp.quit()

        string.seek(0)
        f = gzip.GzipFile(fileobj=string)
        self._add_file(f)
        string.close()
        f.close()

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance.
        """
        null = Q_(float("nan"),self.source_unit)
        return self.data.get(day.strftime("%Y%m%d"),null).to(unit).magnitude

    def _add_file(self,f):
        for line in f.readlines()[1:]:
            columns=line.split()
            date_str = columns[2].decode('utf-8')
            temp = Q_(float(columns[3]),self.source_unit)
            self.data[date_str] = temp
            temp_C = temp.to(ureg.degC).magnitude
            dat = datetime.strptime(date_str,"%Y%m%d").date()
            self.update_cache(temp_C,dat)

class ISDWeatherSource(WeatherSourceBase):
    def __init__(self,station_id,start_year,end_year):
        super(ISDWeatherSource,self).__init__()
        if len(station_id) == 6:
            # given station id is the six digit code, so need to get full name
            with resource_stream('eemeter.resources','GSOD-ISD_station_index.json') as f:
                station_index = json.loads(f.read().decode("utf-8"))
            # take first station in list
            potential_station_ids = station_index[station_id]
        else:
            # otherwise, just use the given id
            potential_station_ids = [station_id]
        ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
        ftp.login()
        for year in range(start_year,end_year + 1):
            string = BytesIO()
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

    def get_period_average_temperature(self,period,unit):
        """Gets the average temperature during a particular DatetimePeriod
        instance. Resolution limit: hourly.
        """
        avg_temps = []
        null = Q_(float("nan"),self.source_unit)
        n_hours = period.timedelta.days * 24 + period.timedelta.seconds // 3600
        for hours in range(n_hours):
            hour = period.start + timedelta(seconds=hours*3600)
            hourly = self.data.get(hour.strftime("%Y%m%d%H"),null).to(unit).magnitude
            avg_temps.append(hourly)
        # mask nans
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        null = Q_(float("nan"),self.source_unit)
        day_str = day.strftime("%Y%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self.data.get("{}{:02d}".format(day_str,i),null).to(unit).magnitude
            avg_temps.append(hourly)
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
            if line[87:92].decode('utf-8') == "+9999":
                air_temperature = Q_(float("nan"),self.source_unit)
            else:
                air_temperature = Q_(float(line[87:92]) / 10, self.source_unit)
            self.data[line[15:25].decode('utf-8')] = air_temperature

class TMY3WeatherSource(WeatherSourceBase):
    def __init__(self,station_id):
        super(TMY3WeatherSource,self).__init__()
        self.station_id = station_id
        r = requests.get("http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/{}TYA.CSV".format(station_id))

        for line in r.text.splitlines()[3:]:
            row = line.split(",")
            date_string = row[0][0:2] + row[0][3:5] + row[1][0:2] # MMDDHH
            self.data[date_string] = Q_(float(row[31]),self.source_unit)

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        null = Q_(float("nan"),self.source_unit)
        day_str = day.strftime("%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self.data.get("{}{:02d}".format(day_str,i),null).to(unit).magnitude
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def annual_daily_temperatures(self,unit):
        """Returns a list of daily temperature normals for a typical
        meteorological year.
        """
        null = Q_(float("nan"),self.source_unit)
        start_day = datetime(2012,1,1)
        temps = []
        for days in range(365):
            day = start_day + timedelta(days=days)
            day_temps = []
            for hour in range(24):
                time = day + timedelta(seconds=hour*3600)
                temp = self.data.get(time.strftime("%m%d%H"),null).to(unit).magnitude
                day_temps.append(temp)
            day_data = np.array(day_temps)
            masked_data = np.ma.masked_array(day_data,np.isnan(day_data))
            # wrap in array for compatibility with model input format
            temps.append(np.array([np.mean(masked_data)]))
        return np.array(temps)

class WeatherUndergroundWeatherSource(WeatherSourceBase):
    def __init__(self,zipcode,start,end,api_key):
        super(WeatherUndergroundWeatherSource,self).__init__()
        self.source_unit = ureg.degF
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

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance.
        """
        null = Q_(float("nan"),self.source_unit)
        return self.data.get(day.strftime("%Y%m%d"),null).to(unit).magnitude

    def _get_query_data(self,query):
        for day in requests.get(query).json()["history"]["dailysummary"]:
            date_string = day["date"]["year"] + day["date"]["mon"] + \
                    day["date"]["mday"]
            data = Q_(int(day["meantempi"]),self.source_unit)
            self.data[date_string] = data

def haversine(lat1,lng1,lat2,lng2):
    """ Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])

    # haversine formula
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def lat_lng_to_tmy3(lat,lng):
    """Return the closest TMY3 weather station id (USAF) using latitude and
    longitude coordinates.
    """
    with resource_stream('eemeter.resources','tmy3_to_lat_lng.json') as f:
        index = json.loads(f.read().decode("utf-8"))
    dists = []
    index_list = [i for i in index.items()]
    for station,(stat_lat,stat_lng) in index_list:
        dists.append(haversine(lat,lng,stat_lat,stat_lng))
    return index_list[np.argmin(dists)][0]

def lat_lng_to_zipcode(lat,lng):
    """Return the closest ZIP code using latitude and
    longitude coordinates.
    """
    with resource_stream('eemeter.resources','zipcode_to_lat_lng.json') as f:
        index = json.loads(f.read().decode("utf-8"))
    dists = []
    index_list = [i for i in index.items()]
    for zipcode,(zip_lat,zip_lng) in index_list:
        dists.append(haversine(lat,lng,zip_lat,zip_lng))
    return index_list[np.argmin(dists)][0]

def tmy3_to_lat_lng(station):
    """Return the latitude and longitude coordinates of the given station.
    """
    with resource_stream('eemeter.resources','tmy3_to_lat_lng.json') as f:
        index = json.loads(f.read().decode("utf-8"))
    return index.get(station)

def tmy3_to_zipcode(station):
    """Return the nearest zipcode to the station by latitude and longitude
    centroid. (Note: Not always the same as finding the containing ZIP code
    area)
    """
    with resource_stream('eemeter.resources','tmy3_to_zipcode.json') as f:
        index = json.loads(f.read().decode("utf-8"))
    return index.get(station)

def zipcode_to_lat_lng(zipcode):
    """Return the latitude and longitude centroid of a particular ZIP code.
    """
    with resource_stream('eemeter.resources','zipcode_to_lat_lng.json') as f:
        index = json.loads(f.read().decode("utf-8"))
    return index.get(zipcode)

def zipcode_to_tmy3(zipcode):
    """Return the nearest TMY3 station (by latitude and longitude centroid) of
    the ZIP code.
    """
    with resource_stream('eemeter.resources','zipcode_to_tmy3.json') as f:
        index = json.loads(f.read().decode("utf-8"))
    return index.get(zipcode)
