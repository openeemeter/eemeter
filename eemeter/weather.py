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

conn = None

try:
    from sqlalchemy import create_engine
    from sqlalchemy import Table, Column, MetaData, Integer, String, Float, Date, DateTime, ForeignKey
    from sqlalchemy.sql import select
    from sqlalchemy.orm.exc import NoResultFound
    from sqlalchemy import extract

    metadata = MetaData()

    weather_stations = Table('weatherstation', metadata,
        Column('id', Integer, primary_key=True),
        Column('usaf_id', String(20)),
    )

    hourly_average_temperatures = Table('hourlyaveragetemperature', metadata,
        Column('id', Integer, primary_key=True),
        Column('weatherstation_id', None, ForeignKey('weatherstation.id')),
        Column('temp_C', Float),
        Column('dt', DateTime),
    )

    daily_average_temperatures = Table('dailyaveragetemperature', metadata,
        Column('id', Integer, primary_key=True),
        Column('weatherstation_id', None, ForeignKey('weatherstation.id')),
        Column('temp_C', Float),
        Column('dt', DateTime),
    )

    hourly_temperature_normals = Table('hourlytemperaturenormal', metadata,
        Column('id', Integer, primary_key=True),
        Column('weatherstation_id', None, ForeignKey('weatherstation.id')),
        Column('temp_C', Float),
        Column('dt', DateTime),
    )

    daily_temperature_normals = Table('dailytemperaturenormal', metadata,
        Column('id', Integer, primary_key=True),
        Column('weatherstation_id', None, ForeignKey('weatherstation.id')),
        Column('temp_C', Float),
        Column('dt', DateTime),
    )
except ImportError:
    warnings.warn("Weather cache disabled. To use, please install sqlalchemy.")

def initialize_cache():
    cache_db_url = os.environ.get("EEMETER_WEATHER_CACHE_DATABASE_URL")
    if cache_db_url is None:
        warnings.warn("Weather cache disabled. To use, please set the"\
                " EEMETER_WEATHER_CACHE_DATABASE_URL environment variable.")
        return None
    engine = create_engine(cache_db_url)
    metadata.create_all(engine)
    conn = engine.connect()
    return conn

class WeatherSourceBase(object):

    def __init__(self,*args,**kwargs):
        super(WeatherSourceBase, self).__init__(*args, **kwargs)
        self.data = {}
        self.station_id = None
        self._internal_unit = "degF"

    def average_temperature(self,periods,unit):
        """The average temperatures during each period as calculated by taking
        the mean of all available daily average temperatures during that
        period.

        Parameters
        ----------
        periods : [list of] eemeter.consumption.DatetimePeriod
            Time periods over which temperatures will be aggregated. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which average temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of average temperatures observed during each period. If a
            single datetime period is given, a single temperature will be
            returned as a float.
        """
        try:
            temps = []
            for period in periods:
                temps.append(self._period_average_temperature(period,unit=None))
            return self._unit_convert(np.array(temps),unit)
        except TypeError:
            return self._period_average_temperature(periods,unit)

    def _period_average_temperature(self,period,unit):
        """The average temperatures during the period as calculated by taking
        the mean of all available daily average temperatures during the period.

        Parameters
        ----------
        period : eemeter.consumption.DatetimePeriod
            Time period over which temperatures will be aggregated.
        unit : {"degC", "degF"}
            The unit in which average temperature should be returned.

        Returns
        -------
        out : float
            Average temperature observed during the period.
        """
        temps = self._period_daily_temperatures(period,unit)
        return np.mean(temps)

    def daily_temperatures(self,periods,unit):
        """The daily average temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.consumption.DatetimePeriod
            Time periods over which temperatures will be aggregated. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of arrays of average daily temperatures observed during each
            period. Note: array is not guaranteed to be rectangular. If a
            single datetime period is given, a single numpy array of
            temperatures will be returned.
        """
        try:
            temps = []
            for period in periods:
                temps.append(self._period_daily_temperatures(period,unit=None))
            if unit is None:
                return np.array(temps)
            else:
                return self._unit_convert(np.array(temps),unit)
        except TypeError:
            return self._period_daily_temperatures(periods,unit)

    def _period_daily_temperatures(self,period,unit):
        """The daily average temperatures for a particular period.

        Parameters
        ----------
        period : eemeter.consumption.DatetimePeriod
            Time period over which temperatures will be aggregated.
        unit : {"degC", "degF"}
            The unit in which daily average temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Daily average temperatures observed during the period.
        """
        temps = []
        for days in range(period.timedelta.days):
            dt = period.start + timedelta(days=days)
            temps.append(self.datetime_average_temperature(dt,unit))
        return np.array(temps)

    def datetime_average_temperature(self,dt,unit):
        """The daily average temperatures for a particular period.

        Parameters
        ----------
        dt : datetime.datetime or datetime.date
            The date for which to find or calculate the average temperature.
        unit : {"degC", "degF"}
            The unit in which the daily average temperature should be returned.

        Returns
        -------
        out : np.ndarray
            Average temperature observed.
        """
        internal_unit_temp = self.internal_unit_datetime_average_temperature(dt)
        if unit is None:
            return internal_unit_temp
        else:
            return self._unit_convert(internal_unit_temp,unit)

    def internal_unit_datetime_average_temperature(self,dt):
        """Should return the average temperature stored in the weather source's
        internal units. Must be implemented by extending classes.
        """
        raise NotImplementedError

    def hdd(self,periods,unit,base,per_day=False):
        """The total heating degree days observed during each time period.

        Parameters
        ----------
        periods : [list of] eemeter.consumption.DatetimePeriod
            Time periods over which heating degree days will be calculated and
            collected. A single period may be supplied.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the heating degree day
        per_day : bool
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : np.ndarray
            Array of heating degree days observed during each time period.
        """
        try:
            hdds = []
            for period in periods:
                hdds.append(self._period_hdd(period,unit,base,per_day))
            return np.array(hdds)
        except TypeError:
            # periods is not iterable
            return self._period_hdd(periods,unit,base,per_day)

    def _period_hdd(self,period,unit,base,per_day):
        """The total heating degree days observed during a particular
        time period.

        Parameters
        ----------
        period : eemeter.consumption.DatetimePeriod
            Time period over which heating degree days will be summed.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the heating degree day
        per_day : bool
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : float
            Total heating degree days observed during the time period.
        """
        temps = self._period_daily_temperatures(period,unit)
        total_hdd = np.sum(np.maximum(base - temps,0))
        if per_day:
            n_days = period.timedelta.days
            return total_hdd / n_days
        else:
            return total_hdd

    def cdd(self,periods,unit,base,per_day=False):
        """The total cooling degree days observed during each time period.

        Parameters
        ----------
        periods : list of eemeter.consumption.DatetimePeriod objects
            Time periods over which cooling degree days will be calculated and
            collected. A single DatetimePeriod may be given.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the cooling degree day
        per_day : bool
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : np.ndarray
            Array of cooling degree days observed during each time period.
        """
        try:
            cdds = []
            for period in periods:
                cdds.append(self._period_cdd(period,unit,base,per_day))
            return np.array(cdds)
        except TypeError:
            # periods is not iterable
            return self._period_cdd(periods,unit,base,per_day)

    def _period_cdd(self,period,unit,base,per_day):
        """The total cooling degree days observed during a particular
        time period.

        Parameters
        ----------
        period : eemeter.consumption.DatetimePeriod
            Time period over which cooling degree days will be summed.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the cooling degree day
        per_day : bool
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : float
            Total cooling degree days observed during the time period.
        """
        temps = self._period_daily_temperatures(period,unit)
        total_cdd = np.sum(np.maximum(temps - base,0))
        if per_day:
            n_days = period.timedelta.days
            return total_cdd / n_days
        else:
            return total_cdd

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

    def __init__(self,*args,**kwargs):
        super(CachedDataMixin,self).__init__(*args,**kwargs)
        global conn
        if not conn:
            try:
                conn = initialize_cache()
            except NameError:
                conn = None

        self.conn = conn

    def init_temperature_data(self):
        """Pulls all cached weather data into memory.
        """
        self.get_weather_station()
        if self.weather_station_pk:
            date_format = self.get_date_format()
            for t in self.get_temperature_set():
                temp_C = t.temp_C
                if temp_C is None:
                    temp_C = float('nan')
                if self._internal_unit == "degC":
                    self.data[t.dt.strftime(date_format)] = temp_C
                elif self._internal_unit == "degF":
                    self.data[t.dt.strftime(date_format)] = self._degC_to_degF(temp_C)

    def get_temperature_table(self):
        """Returns the SQLAlchemy database class used for caching.

        E.g. `return hourly_average_temperatures`
        """
        raise NotImplementedError

    def get_temperature_set(self):
        """Returns the set of all database temperature objects.
        """
        temperature_table = self.get_temperature_table()
        s = select([temperature_table]).where(temperature_table.c.weatherstation_id == self.weather_station_pk)
        result = self.conn.execute(s)
        return result

    def get_date_format(self):
        """Returns the date format for fetching and storing temperature objects
        in memory.

        E.g. `return "%Y%m%d"`
        """
        raise NotImplementedError

    def get_weather_station(self):
        """Sets the weather_station_pk attribute using the db connection and
        self.station_id, if available.
        """
        if self.conn:
            s = select([weather_stations.c.id]).where(weather_stations.c.usaf_id == self.station_id)
            result = self.conn.execute(s)
            row = result.fetchone()
            if row:
                self.weather_station_pk = row['id']
            else:
                ins = weather_stations.insert().values(usaf_id=self.station_id)
                result = self.conn.execute(ins)
                self.weather_station_pk = result.inserted_primary_key[0]
            result.close()
        else:
            self.weather_station_pk = None

    def update_cache(self,records):
        """If caching is enabled, update the cache for the given data.
        """
        # add current weather station to each temperature to be inserted.
        for r in records:
            r["weatherstation_id"] = self.weather_station_pk

        # figure out which dates need to be overwritten
        dates = set([d["dt"] for d in records])

        if self.conn:

            # delete existing records (if necessary)
            all_temps = self.get_temperature_set()
            old_records = [t for t in all_temps if t["dt"] in dates]
            for record in old_records:
                self.conn.execute(record.delete())

            # insert new records
            self.conn.execute(self.get_temperature_table().insert(), records)

    def has_cached_data_for_year(self,year):
        """Return True if data for the given year is in the cache.
        """
        if self.conn:
            temps = self.get_temperature_table()
            s = select([temps.c.dt]).where(extract('year', temps.c.dt) == year)
            result = self.conn.execute(s)
            n_results = len(result.fetchall())
            return n_results > 1000
        return False

class HourlyTemperatureNormalCachedDataMixin(CachedDataMixin):

    def get_temperature_table(self):
        """Returns the SQLAlchemy database table used for caching.
        """
        return hourly_temperature_normals

    def get_date_format(self):
        """Returns the date format for fetching and storing temperature objects
        in memory.
        """
        return "%m%d%H"

class DailyTemperatureNormalCachedDataMixin(CachedDataMixin):

    def get_temperature_table(self):
        """Returns the SQLAlchemy database table used for caching.
        """
        return daily_temperature_normals

    def get_date_format(self):
        """Returns the date format for fetching and storing temperature objects
        in memory.
        """
        return "%m%d"

class HourlyAverageTemperatureCachedDataMixin(CachedDataMixin):

    def get_temperature_table(self):
        """Returns the SQLAlchemy database table used for caching.
        """
        return hourly_average_temperatures

    def get_date_format(self):
        """Returns the date format for fetching and storing temperature objects
        in memory.
        """
        return "%Y%m%d%H"

class DailyAverageTemperatureCachedDataMixin(CachedDataMixin):

    def get_temperature_table(self):
        """Returns the SQLAlchemy database table used for caching.
        """
        return daily_average_temperatures

    def get_date_format(self):
        """Returns the date format for fetching and storing temperature objects
        in memory.
        """
        return "%Y%m%d"

class GSODWeatherSource(DailyAverageTemperatureCachedDataMixin,WeatherSourceBase):
    def __init__(self,station_id,start_year,end_year):
        super(GSODWeatherSource,self).__init__()
        self.station_id = station_id[:6]
        self.init_temperature_data() # load cached data

        if self.data == {}:
            for year in range(start_year,end_year + 1):
                self._fetch_year(year)
        else:
            for year in range(start_year,end_year + 1):
                one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
                if (year == datetime.now().year and not self.data.get(one_week_ago)) \
                        or not self.has_cached_data_for_year(year):
                    self._fetch_year(year)

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

    def internal_unit_datetime_average_temperature(self,dt):
        """Returns the average temperature on the given day. `dt` can be
        either a python `date` or a python `datetime` instance. Temperature is
        given in the units in which it is internally stored.
        """
        return self.data.get(dt.strftime("%Y%m%d"),float("nan"))

    def _add_file(self,f):
        records = []
        for line in f.readlines()[1:]:
            columns=line.split()
            date_str = columns[2].decode('utf-8')
            temp = float(columns[3])
            self.data[date_str] = temp
            temp_C = self._degF_to_degC(temp)
            dt = datetime.strptime(date_str,"%Y%m%d").date()
            records.append({"temp_C": temp_C, "dt": dt})
        self.update_cache(records)

class ISDWeatherSource(WeatherSourceBase,HourlyAverageTemperatureCachedDataMixin):
    def __init__(self,station_id,start_year,end_year):
        super(ISDWeatherSource,self).__init__()
        self.station_id = station_id[:6]
        self.init_temperature_data() # load cached data

        if self.data == {}:
            for year in range(start_year,end_year + 1):
                self._fetch_year(year)
        else:
            for year in range(start_year,end_year + 1):
                one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d") + "00"
                if (year == datetime.now().year and not self.data.get(one_week_ago))\
                        or not self.has_cached_data_for_year(year):
                    self._fetch_year(year)

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

    def internal_unit_datetime_average_temperature(self,dt):
        """Returns the average temperature on the given datetime. `dt` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        dt_str = dt.strftime("%Y%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self.data.get("{}{:02d}".format(dt_str,i),float("nan"))
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

    def hourly_temperatures(self,periods,unit):
        """The hourly observed temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.consumption.DatetimePeriod
            Time periods over which temperatures will be collected. A single
            datetime period may be given.
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            Array of arrays of observed_daily temperatures observed during each
            period. Note: array is not guaranteed to be rectangular. If a
            single datetime period is given, a single numpy array of
            temperatures will be returned.
        """
        try:
            temps = []
            for period in periods:
                temps.append(self._period_hourly_temperatures(period,unit=None))
            if unit is None:
                return np.array(temps)
            else:
                return self._unit_convert(np.array(temps),unit)
        except TypeError:
            return self._period_hourly_temperatures(periods,unit)

    def _period_hourly_temperatures(self,period,unit):
        temps = []
        for seconds in range(0,int(period.timedelta.total_seconds()),3600):
            dt = period.start + timedelta(seconds=seconds)
            temps.append(self.datetime_hourly_temperature(dt,unit))
        return np.array(temps)

    def datetime_hourly_temperature(self,dt,unit):
        internal_unit_temp = self.internal_unit_datetime_hourly_temperature(dt)
        if unit is None:
            return internal_unit_temp
        else:
            return self._unit_convert(internal_unit_temp,unit)

    def internal_unit_datetime_hourly_temperature(self,dt):
        """Returns the hourly temperature on the given datetime. `dt` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        dt_str = dt.strftime("%Y%m%d%H")
        hourly_temp = self.data.get(dt_str,float("nan"))
        return hourly_temp

    def _add_file(self,f):
        records = []
        for line in f.readlines():
            if line[87:92].decode('utf-8') == "+9999":
                temp_C = float("nan")
            else:
                temp_C = float(line[87:92]) / 10
            date_str = line[15:25].decode('utf-8')
            self.data[date_str] = self._degC_to_degF(temp_C)
            dt = datetime.strptime(date_str,"%Y%m%d%H")
            records.append({"temp_C": temp_C, "dt": dt})
        self.update_cache(records)

class WeatherNormalMixin(object):
    def annual_daily_temperatures(self,unit):
        """Returns a list of daily temperature normals for a typical
        meteorological year.
        """
        periods = [DatetimePeriod(start=datetime(2013,1,1),end=datetime(2014,1,1))]
        return self.daily_temperatures(periods,unit)

class TMY3WeatherSource(WeatherSourceBase,WeatherNormalMixin,HourlyTemperatureNormalCachedDataMixin):
    def __init__(self,station_id):
        super(TMY3WeatherSource,self).__init__()
        self.station_id = station_id
        self._internal_unit = "degC"
        self.init_temperature_data() # load cached data

        n_temp_normals = len(self.data.items())
        if n_temp_normals < 364 * 24: #missing more than a day of data
            self._fetch_data()

    def _fetch_data(self):
        r = requests.get("http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/{}TYA.CSV".format(self.station_id))

        records = []
        for line in r.text.splitlines()[3:]:
            row = line.split(",")
            date_string = "{}{}{}{:02d}".format(row[0][6:10], row[0][0:2],
                                                row[0][3:5], int(row[1][0:2]) - 1) # YYYYMMDDHH
            temp_C = float(row[31])
            self.data[date_string[4:]] = temp_C # skip year in date string
            dt = datetime.strptime(date_string,"%Y%m%d%H")
            records.append({"temp_C": temp_C, "dt": dt})
        self.update_cache(records)

    def internal_unit_datetime_average_temperature(self,dt):
        """Returns the temperature normal on the given datetime. `dt` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        dt_str = dt.strftime("%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self.data.get("{}{:02d}".format(dt_str,i),float('nan'))
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

class CZ2010WeatherSource(WeatherSourceBase,WeatherNormalMixin):
    def __init__(self,filepath):
        super(CZ2010WeatherSource,self).__init__()
        self.filepath = filepath

        self._fetch_data()

    def _fetch_data(self):
        with open(self.filepath, 'r') as f:
            text = f.read()
            for line in text.splitlines()[2:]:
                row = line.split(",")
                date_string = "{}{}{}{:02d}".format(row[0][6:10],
                                                    row[0][0:2],
                                                    row[0][3:5],
                                                    int(row[1][0:2]) - 1)
                temp_C = float(row[31])
                self.data[date_string[4:]] = self._degC_to_degF(temp_C) # skip year in date string

    def internal_unit_datetime_average_temperature(self,dt):
        """Returns the temperature normal on the given datetime. `dt` can be
        either a python `date` or a python `datetime` instance. Calculated by
        averaging hourly temperatures for the given day.
        """
        dt_str = dt.strftime("%m%d")
        avg_temps = []
        for i in range(24):
            hourly = self.data.get("{}{:02d}".format(dt_str,i),float('nan'))
            avg_temps.append(hourly)
        data = np.array(avg_temps)
        masked_data = np.ma.masked_array(data,np.isnan(data))
        return np.mean(masked_data)

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

    def internal_unit_datetime_average_temperature(self,dt):
        """Returns the average temperature on the given datetime. `dt` can be
        either a python `date` or a python `datetime` instance.
        """
        return self.data.get(dt.strftime("%Y%m%d"),float('nan'))

    def _get_query_data(self,query):
        for day in requests.get(query).json()["history"]["dailysummary"]:
            date_string = day["date"]["year"] + day["date"]["mon"] + \
                    day["date"]["mday"]
            self.data[date_string] = float(day["meantempi"])
