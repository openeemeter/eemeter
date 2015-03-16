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
from pkg_resources import resource_stream

from eemeter.consumption import DatetimePeriod

tmy3_to_lat_lng_index = None
tmy3_to_zipcode_index = None
zipcode_to_lat_lng_index = None
zipcode_to_tmy3_index = None

Session = None

try:
    from sqlalchemy import create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.orm import relationship, backref
    from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey
    from sqlalchemy.orm.exc import NoResultFound

    Base = declarative_base()

    class WeatherStation(Base):
        __tablename__ = 'weatherstation'

        id = Column(Integer, primary_key=True)
        usaf_id = Column(String)
        hourly_temperature_normals = relationship("HourlyTemperatureNormal", order_by="HourlyTemperatureNormal.date", backref="weatherstation")
        daily_temperature_normals = relationship("DailyTemperatureNormal", order_by="DailyTemperatureNormal.date", backref="weatherstation")
        hourly_average_temperatures = relationship("HourlyAverageTemperature", order_by="HourlyAverageTemperature.date", backref="weatherstation")
        daily_average_temperatures = relationship("DailyAverageTemperature", order_by="DailyAverageTemperature.date", backref="weatherstation")

        def __repr__(self):
               return "<WeatherStation('{}')".format(self.usaf_id)

    class HourlyAverageTemperature(Base):
        __tablename__ = 'hourlyaveragetemperature'

        id = Column(Integer, primary_key=True)
        weatherstation_id = Column(Integer, ForeignKey('weatherstation.id'))
        temp_C = Column(Float)
        date = Column(DateTime)

        def __repr__(self):
               return "<HourlyAverageTemperature('{}', {}, {})>".format(
                                    self.weatherstation.usaf_id, self.temp_C, self.date)

    class DailyAverageTemperature(Base):
        __tablename__ = 'dailyaveragetemperature'

        id = Column(Integer, primary_key=True)
        weatherstation_id = Column(Integer, ForeignKey('weatherstation.id'))
        temp_C = Column(Float)
        date = Column(Date)

        def __repr__(self):
               return "<DailyAverageTemperature('{}', {}, {})>".format(
                                    self.weatherstation.usaf_id, self.temp_C, self.date)

    class HourlyTemperatureNormal(Base):
        __tablename__ = 'hourlytemperaturenormal'

        id = Column(Integer, primary_key=True)
        weatherstation_id = Column(Integer, ForeignKey('weatherstation.id'))
        temp_C = Column(Float)
        date = Column(DateTime)

        def __repr__(self):
               return "<HourlyTemperatureNormal('{}', {}, {})>".format(
                                    self.weatherstation.usaf_id, self.temp_C, self.date)

    class DailyTemperatureNormal(Base):
        __tablename__ = 'dailytemperaturenormal'

        id = Column(Integer, primary_key=True)
        weatherstation_id = Column(Integer, ForeignKey('weatherstation.id'))
        temp_C = Column(Float)
        date = Column(Date)

        def __repr__(self):
               return "<DailyTemperatureNormal('{}', {}, {})>".format(
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
        self.station_id = None
        self._internal_unit = "degF"

    def get_average_temperature(self,periods,unit):
        """Returns a list of average temperatures of each DatetimePeriod in
        the given unit (usually "degF" or "degC").
        """
        temps = []
        for period in periods:
            temps.append(self.get_period_average_temperature(period,unit=None))
        return self._unit_convert(np.array(temps),unit)

    def get_period_average_temperature(self,period,unit):
        """Returns the average temperature during the duration of a single
        DatetimePeriod instance as calculated by taking the mean of daily average
        temperatures.
        """
        temps = self.get_period_daily_temperatures(period,unit)
        return np.mean(temps)

    def get_daily_temperatures(self,periods,unit):
        """Returns, for each period, a list of average daily temperatures
        observed during the duration of that period. Return value
        is a list of lists of temperatures.
        """
        temps = []
        for period in periods:
            temps.append(self.get_period_daily_temperatures(period,unit=None))
        if unit is None:
            return np.array(temps)
        else:
            return self._unit_convert(np.array(temps),unit)

    def get_period_daily_temperatures(self,period,unit):
        """Returns, for a particular period instance, a list of average
        daily temperatures observed during the duration of that period
        period. Result is a list of temperatures.
        """
        temps = []
        for days in range(period.timedelta.days):
            day = period.start + timedelta(days=days)
            temps.append(self.get_daily_average_temperature(day,unit))
        if unit is None:
            return np.array(temps)
        else:
            return self._unit_convert(np.array(temps),unit)

    def get_daily_average_temperature(self,day,unit):
        """Returns the average temperature of the given day in correct units
        """
        internal_unit_temp = self.get_internal_unit_daily_average_temperature(day)
        if unit is None:
            return internal_unit_temp
        else:
            return self._unit_convert(internal_unit_temp,unit)

    def get_internal_unit_daily_average_temperature(self,day):
        """Should return the average temperature stored in the weather source's
        internal units. Must be implemented by extending classes.
        """
        raise NotImplementedError

    def get_hdd(self,periods,unit,base):
        """Returns, for each period, the total heating degree days
        observed during the period.
        """
        hdds = []
        for period in periods:
            hdds.append(self.get_period_hdd(period,unit,base))
        return np.array(hdds)

    def get_hdd_per_day(self,periods,unit,base):
        """Returns, for each period, the total heating degree days
        observed during the period.
        """
        hdds_per_day = []
        for period in periods:
            hdds_per_day.append(self.get_period_hdd_per_day(period,unit,base))
        return np.array(hdds_per_day)

    def get_period_hdd(self,period,unit,base):
        """Returns the total heating degree days observed during the
        period.
        """
        temps = self.get_period_daily_temperatures(period,unit)
        return np.sum(np.maximum(base - temps,0))

    def get_period_hdd_per_day(self,period,unit,base):
        """Returns the total heating degree days observed during the
        period.
        """
        period_hdd = self.get_period_hdd(period,unit,base)
        n_days = period.timedelta.days
        return period_hdd / n_days

    def get_cdd(self,periods,unit,base):
        """Returns, for each period, the total cooling degree days
        observed during the period.
        """
        cdds = []
        for period in periods:
            cdds.append(self.get_period_cdd(period,unit,base))
        return np.array(cdds)

    def get_cdd_per_day(self,periods,unit,base):
        """Returns, for each period, the total cooling degree days
        observed during the period.
        """
        cdds_per_day = []
        for period in periods:
            cdds_per_day.append(self.get_period_cdd_per_day(period,unit,base))
        return np.array(cdds_per_day)


    def get_period_cdd(self,period,unit,base):
        """Returns the total cooling degree days observed during the
        period.
        """
        temps = self.get_period_daily_temperatures(period,unit)
        return np.sum(np.maximum(temps - base,0))

    def get_period_cdd_per_day(self,period,unit,base):
        """Returns the total cooling degree days observed during the
        period.
        """
        period_cdd = self.get_period_cdd(period,unit,base)
        n_days = period.timedelta.days
        return period_cdd / n_days

    def _unit_convert(self,temps_array,unit):
        """Returns an array converted to the correct units.
        """
        if unit == self._internal_unit:
            return temps_array
        else:
            if unit == "degC":
                return self._degF_to_degC(temps_array)
            elif unit == "degF":
                return self._degC_to_degF(temps_array)
            else:
                message = "Unit not recognized ({}). Should be one of 'degC' or 'degF'".format(unit)
                raise ValueError(message)

    @staticmethod
    def _degC_to_degF(temp_C):
        """Returns temperature(s) in degrees Fahrenheit given a temperature (or
        array of temperatures) in degrees Celsius.
        """
        return 1.8*temp_C + 32.

    @staticmethod
    def _degF_to_degC(temp_F):
        """Returns temperature(s) in degrees Celsius given a temperature (or
        array of temperatures) in degrees Fahrenheit.
        """
        return (5./9.) * (temp_F - 32.)

class CachedDataMixin(object):

    def init_temperature_data(self):
        """Pulls all cached weather data into memory.
        """
        self.get_weather_station()
        if self.weather_station:
            date_format = self.get_date_format()
            if self._internal_unit == "degC":
                for t in self.get_temperature_set():
                    self.data[t.date.strftime(date_format)] = t.temp_C
            elif self._internal_unit == "degF":
                for t in self.get_temperature_set():
                    self.data[t.date.strftime(date_format)] = self._degC_to_degF(t.temp_C)

    def get_temperature_class(self):
        """Returns the SQLAlchemy database class used for caching.

        E.g. `return Temperature`
        """
        raise NotImplementedError

    def get_temperature_set(self):
        """Returns the set of all database temperature objects.

        E.g. `return self.weather_station.temperatures`
        """
        raise NotImplementedError

    def get_date_format(self):
        """Returns the date format for fetching and storing temperature objects
        in memory.

        E.g. `return "%Y%m%d"`
        """
        raise NotImplementedError

    def get_weather_station(self):
        """Sets the weather_station attribute using the db session and
        self.station_id, if available.
        """
        if self.session:
            try:
                self.weather_station = self.session.query(WeatherStation).filter(WeatherStation.usaf_id == self.station_id).one()
            except NoResultFound:
                self.weather_station = WeatherStation(usaf_id=self.station_id)
                self.session.add(self.weather_station)
                self.session.commit()
        else:
            self.weather_station = None

    def update_cache(self,temp_C,date,overwrite=True):
        """If cacheing is enabled, store the given temp, overwriting if necessary.

        Warning: Slow! (TODO: speed this up)
        """
        if self.session:
            temperature_class = self.get_temperature_class()
            temps_query = self.session.query(temperature_class)\
                        .filter(temperature_class.weatherstation == self.weather_station)\
                        .filter(temperature_class.date==date)

            temps = temps_query.all()
            if overwrite:
                for t in temps:
                    self.session.delete(t)
                temps = []
            if temps == []:
                t = temperature_class(weatherstation=self.weather_station,temp_C=temp_C,date=date)
                self.session.add(t)

class HourlyTemperatureNormalCachedDataMixin(CachedDataMixin):

    def get_temperature_class(self):
        return HourlyTemperatureNormal

    def get_temperature_set(self):
        return self.weather_station.hourly_temperature_normals

    def get_date_format(self):
        return "%m%d%H"

class DailyTemperatureNormalCachedDataMixin(CachedDataMixin):

    def get_temperature_class(self):
        return DailyTemperatureNormal

    def get_temperature_set(self):
        return self.weather_station.daily_temperature_normals

    def get_date_format(self):
        return "%m%d"

class HourlyAverageTemperatureCachedDataMixin(CachedDataMixin):

    def get_temperature_class(self):
        return HourlyAverageTemperature

    def get_temperature_set(self):
        return self.weather_station.hourly_average_temperatures

    def get_date_format(self):
        return "%Y%m%d%H"

class DailyAverageTemperatureCachedDataMixin(CachedDataMixin):

    def get_temperature_class(self):
        return DailyAverageTemperature

    def get_temperature_set(self):
        return self.weather_station.daily_average_temperatures

    def get_date_format(self):
        return "%Y%m%d"

class GSODWeatherSource(WeatherSourceBase,DailyAverageTemperatureCachedDataMixin):
    def __init__(self,station_id,start_year,end_year):
        super(GSODWeatherSource,self).__init__()
        self.station_id = station_id[:6]
        self.init_temperature_data()

        for days in range((datetime(end_year,12,31) - datetime(start_year,1,1)).days):
            dat = datetime(start_year,1,1) + timedelta(days=days)
            if dat > datetime.now() - timedelta(days=1):
                break
            temp = self.data.get(dat.strftime("%Y%m%d"))
            if temp is None:
                self._fetch_year(dat.year)
        if self.session:
            self.session.close()

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

    def get_internal_unit_daily_average_temperature(self,day):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance. Temperature is
        given in the units in which it is internally stored.
        """
        return self.data.get(day.strftime("%Y%m%d"),float("nan"))

    def _add_file(self,f):
        for line in f.readlines()[1:]:
            columns=line.split()
            date_str = columns[2].decode('utf-8')
            temp = float(columns[3])
            self.data[date_str] = temp
            temp_C = self._degF_to_degC(temp)
            dat = datetime.strptime(date_str,"%Y%m%d").date()
            self.update_cache(temp_C,dat)
        if self.session:
            self.session.commit()

class ISDWeatherSource(WeatherSourceBase,HourlyAverageTemperatureCachedDataMixin):
    def __init__(self,station_id,start_year,end_year):
        super(ISDWeatherSource,self).__init__()
        self.station_id = station_id[:6]
        self.init_temperature_data()

        if self.data == {}:
            for year in range(start_year,end_year + 1):
                self._fetch_year(year)
        else:
            for year in range(start_year,end_year + 1):
                if year == datetime.now().year and not self.data.get(datetime.now().strftime("%Y%m%d") + "00"):
                    self._fetch_year(datetime.now().year)
                else:
                    temps = []
                    for days in range(365):
                        dat = datetime(year,1,1) + timedelta(days=days)
                        temps.append(self.data.get(dat.strftime("%Y%m%d") + "00"))
                        temps.append(self.data.get(dat.strftime("%Y%m%d") + "01"))
                if len(temps) < 700:
                    self._fetch_year(dat.year)
        if self.session:
            self.session.close()

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

    def get_internal_unit_daily_average_temperature(self,day):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        day_str = day.strftime("%Y%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self.data.get("{}{:02d}".format(day_str,i),float("nan"))
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def _add_file(self,f):
        for line in f.readlines():
            if line[87:92].decode('utf-8') == "+9999":
                temp_C = float("nan")
            else:
                temp_C = float(line[87:92]) / 10
            date_str = line[15:25].decode('utf-8')
            self.data[date_str] = self._degC_to_degF(temp_C)
            dat = datetime.strptime(date_str,"%Y%m%d%H")
            self.update_cache(temp_C,dat)
        if self.session:
            self.session.commit()

class TMY3WeatherSource(WeatherSourceBase,HourlyTemperatureNormalCachedDataMixin):
    def __init__(self,station_id):
        super(TMY3WeatherSource,self).__init__()
        self.station_id = station_id
        self.init_temperature_data()

        n_temp_normals = len(self.data.items())
        if n_temp_normals < 364 * 24: #missing more than a day of data
            self._fetch_data()

        if self.session:
           self.session.close()

    def _fetch_data(self):
        r = requests.get("http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/{}TYA.CSV".format(self.station_id))

        for line in r.text.splitlines()[3:]:
            row = line.split(",")
            date_string = "{}{}{}{:02d}".format(row[0][6:10], row[0][0:2],
                                                row[0][3:5], int(row[1][0:2]) - 1) # YYYYMMDDHH
            temp_C = float(row[31])
            self.data[date_string[4:]] = self._degC_to_degF(temp_C) # skip year in date string
            dat = datetime.strptime(date_string,"%Y%m%d%H")
            self.update_cache(temp_C,dat)
        if self.session:
            self.session.commit()

    def get_internal_unit_daily_average_temperature(self,day):
        """Returns the temperature normal on the given day. `day` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        day_str = day.strftime("%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self.data.get("{}{:02d}".format(day_str,i),float('nan'))
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def annual_daily_temperatures(self,unit):
        """Returns a list of daily temperature normals for a typical
        meteorological year.
        """
        periods = [DatetimePeriod(start=datetime(2013,1,1),end=datetime(2014,1,1))]
        return self.get_daily_temperatures(periods,unit)

class WeatherUndergroundWeatherSource(WeatherSourceBase):
    def __init__(self,zipcode,start,end,api_key):
        super(WeatherUndergroundWeatherSource,self).__init__()
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

    def get_internal_unit_daily_average_temperature(self,day):
        """Returns the average temperature on the given day. `day` can be
        either a python `date` or a python `datetime` instance.
        """
        return self.data.get(day.strftime("%Y%m%d"),float('nan'))

    def _get_query_data(self,query):
        for day in requests.get(query).json()["history"]["dailysummary"]:
            date_string = day["date"]["year"] + day["date"]["mon"] + \
                    day["date"]["mday"]
            self.data[date_string] = float(day["meantempi"])

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
    global tmy3_to_lat_lng_index
    if tmy3_to_lat_lng_index is None:
        with resource_stream('eemeter.resources','tmy3_to_lat_lng.json') as f:
            tmy3_to_lat_lng_index = json.loads(f.read().decode("utf-8"))
    dists = []
    index_list = [i for i in tmy3_to_lat_lng_index.items()]
    for station,(stat_lat,stat_lng) in index_list:
        dists.append(haversine(lat,lng,stat_lat,stat_lng))
    return index_list[np.argmin(dists)][0]

def lat_lng_to_zipcode(lat,lng):
    """Return the closest ZIP code using latitude and
    longitude coordinates.
    """
    global zipcode_to_lat_lng_index
    if zipcode_to_lat_lng_index is None:
        with resource_stream('eemeter.resources','zipcode_to_lat_lng.json') as f:
            zipcode_to_lat_lng_index = json.loads(f.read().decode("utf-8"))
    dists = []
    index_list = [i for i in zipcode_to_lat_lng_index.items()]
    for zipcode,(zip_lat,zip_lng) in index_list:
        dists.append(haversine(lat,lng,zip_lat,zip_lng))
    return index_list[np.argmin(dists)][0]

def tmy3_to_lat_lng(station):
    """Return the latitude and longitude coordinates of the given station.
    """
    global tmy3_to_lat_lng_index
    if tmy3_to_lat_lng_index is None:
        with resource_stream('eemeter.resources','tmy3_to_lat_lng.json') as f:
            tmy3_to_lat_lng_index = json.loads(f.read().decode("utf-8"))
    return tmy3_to_lat_lng_index.get(station)

def tmy3_to_zipcode(station):
    """Return the nearest zipcode to the station by latitude and longitude
    centroid. (Note: Not always the same as finding the containing ZIP code
    area)
    """
    global tmy3_to_zipcode_index
    if tmy3_to_zipcode_index is None:
        with resource_stream('eemeter.resources','tmy3_to_zipcode.json') as f:
            tmy3_to_zipcode_index = json.loads(f.read().decode("utf-8"))
    return tmy3_to_zipcode_index.get(station)

def zipcode_to_lat_lng(zipcode):
    """Return the latitude and longitude centroid of a particular ZIP code.
    """
    global zipcode_to_lat_lng_index
    if zipcode_to_lat_lng_index is None:
        with resource_stream('eemeter.resources','zipcode_to_lat_lng.json') as f:
            zipcode_to_lat_lng_index = json.loads(f.read().decode("utf-8"))
    return zipcode_to_lat_lng_index.get(zipcode)

def zipcode_to_tmy3(zipcode):
    """Return the nearest TMY3 station (by latitude and longitude centroid) of
    the ZIP code.
    """
    global zipcode_to_tmy3_index
    if zipcode_to_tmy3_index is None:
        with resource_stream('eemeter.resources','zipcode_to_tmy3.json') as f:
            zipcode_to_tmy3_index = json.loads(f.read().decode("utf-8"))
    return zipcode_to_tmy3_index.get(zipcode)
