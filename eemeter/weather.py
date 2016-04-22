from eemeter.evaluation import Period
from eemeter.location import _load_station_to_lat_lng_index, haversine

from datetime import datetime, date, timedelta
import ftplib
import gzip
from io import BytesIO
import json
import os
from pkg_resources import resource_stream
import warnings

import numpy as np
import pandas as pd
from pandas.core.common import is_list_like
import requests


class NOAAClient(object):

    def __init__(self, n_tries=3):
        self.n_tries = n_tries
        self.ftp = None # lazily load
        self.station_index = None # lazily load

    def _get_ftp_connection(self):
        for _ in range(self.n_tries):
            try:
                ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
                ftp.login()
                return ftp
            except EOFError:
                pass
        raise EOFError

    def _load_station_index(self):
        with resource_stream('eemeter.resources', 'GSOD-ISD_station_index.json') as f:
            return json.loads(f.read().decode("utf-8"))

    def _get_potential_station_ids(self, station):
        if self.station_index is None:
            self.station_index = self._load_station_index()

        if len(station) == 6:
            potential_station_ids = self.station_index[station]
        else:
            potential_station_ids = [station]
        return potential_station_ids

    def _retreive_file_lines(self, filename_format, station, year):
        string = BytesIO()

        if self.ftp is None:
            self.ftp = self._get_ftp_connection()

        for station_id in self._get_potential_station_ids(station):
            filename = filename_format.format(station=station_id, year=year)
            try:
                self.ftp.retrbinary('RETR {}'.format(filename), string.write)
                break
            except (IOError, ftplib.error_perm):
                pass
            except (ftplib.error_temp, EOFError): # Bad connection. attempt to reconnect.
                self.ftp.close()
                self.ftp = self._get_ftp_connection()
                try:
                    self.ftp.retrbinary('RETR {}'.format(filename), string.write)
                    break
                except (IOError, ftplib.error_perm):
                    pass

        string.seek(0)
        f = gzip.GzipFile(fileobj=string)
        lines = f.readlines()
        string.close()
        return lines

    def get_gsod_data(self, station, year):

        filename_format = '/pub/data/gsod/{year}/{station}-{year}.op.gz'
        lines = self._retreive_file_lines(filename_format, station, year)

        days = []
        for line in lines[1:]:
            columns=line.split()
            date_str = columns[2].decode('utf-8')
            temp_F = float(columns[3])
            temp_C = (5./9.) * (temp_F - 32.)
            dt = datetime.strptime(date_str,"%Y%m%d").date()
            days.append({"temp_C": temp_C, "date": dt})

        return days

    def get_isd_data(self, station, year):

        filename_format = '/pub/data/noaa/{year}/{station}-{year}.gz'
        lines = self._retreive_file_lines(filename_format, station, year)

        hours = []
        for line in lines:
            if line[87:92].decode('utf-8') == "+9999":
                temp_C = float("nan")
            else:
                temp_C = float(line[87:92]) / 10.
            date_str = line[15:27].decode('utf-8')
            dt = datetime.strptime(date_str, "%Y%m%d%H%M")
            hours.append({"temp_C": temp_C, "datetime": dt})

        return hours


class TMY3Client(object):

    def __init__(self):
        self.stations = None # lazily load
        self.station_to_lat_lng = None # lazily load

    def _load_stations(self):
        with resource_stream('eemeter.resources', 'tmy3_stations.json') as f:
            return json.loads(f.read().decode("utf-8"))

    def _load_station_locations(self):
        return _load_station_to_lat_lng_index()

    def _find_nearby_station(self, station):
        if self.stations is None:
            self.stations = self._load_stations()
        if self.station_to_lat_lng is None:
            self.station_to_lat_lng = self._load_station_locations()

        try:
            lat, lng = self.station_to_lat_lng[station]
        except KeyError:
            warnings.warn(
                "Could not locate station {}; "
                "nearby station look-up failed".format(station)
            )
            return None
        else:
            index_list = list(self.station_to_lat_lng.items())
            dists = [haversine(lat, lng, stat_lat, stat_lng)
                     for _, (stat_lat, stat_lng) in index_list]

        for dist, (nearby_station, _) in zip(dists, index_list):
            if nearby_station in self.stations:
                warnings.warn("Using station {} instead".format(nearby_station))
                return nearby_station
        return None

    def get_tmy3_data(self, station, station_fallback=True):

        if self.stations is None:
            self.stations = self._load_stations()

        if not station in self.stations:
            warnings.warn(
                "Station {} is not a TMY3 station. "
                "See self.stations for a complete list.".format(station)
                )
            if station_fallback:
                station = self._find_nearby_station(station)
            else:
                station = None

        if station is None:
            return None

        url = "http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/{}TYA.CSV".format(station)
        r = requests.get(url)

        if r.status_code == 200:
            hours = []
            for line in r.text.splitlines()[2:]:
                row = line.split(",")
                year = row[0][6:10]
                month = row[0][0:2]
                day = row[0][3:5]
                hour = int(row[1][0:2]) - 1
                date_string = "{}{}{}{:02d}".format(year, month, day, hour) # YYYYMMDDHH
                dt = datetime.strptime(date_string,"%Y%m%d%H")
                temp_C = float(row[31])
                hours.append({"temp_C": temp_C, "dt": dt})
            return hours
        else:
            warnings.warn("Station {} was not found. Tried url {}.".format(station, url))
            return None


class WeatherSourceBase(object):

    def __init__(self, station):
        self.station = station
        self.tempC = pd.Series(dtype=float)

    @staticmethod
    def _unit_convert(x, unit):
        if unit is None or unit == "degC":
            return x
        elif unit == "degF":
            return 1.8*x + 32
        else:
            message = "Unit not supported ({}). Use 'degF' or 'degC'".format(unit)
            raise NotImplementedError(message)

    def _fetch_period(self, period):
        message = "Inheriting classes must override this method."
        raise NotImplementedError(message)

    def _fetch_datetime(self, dt):
        message = "Inheriting classes must override this method."
        raise NotImplementedError(message)

    def average_temperature(self, periods, unit):
        """The average temperatures during each period as calculated by taking
        the mean of all available daily average temperatures during that
        period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
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

        if is_list_like(periods):
            values = np.array([self._period_average_temperature(p, None) for p in periods])
            return self._unit_convert(values, unit)
        else:
            return self._period_average_temperature(periods, unit)

    def daily_temperatures(self, periods, unit):
        """The daily average temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
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

        if is_list_like(periods):
            values = np.array([self._period_daily_temperatures(p, None) for p in periods])
            return self._unit_convert(values, unit)
        else:
            return self._period_daily_temperatures(periods, unit)

    def hourly_temperatures(self, periods, unit):
        """The hourly observed temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
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

        if is_list_like(periods):
            values = np.array([self._period_hourly_temperatures(p, None) for p in periods])
            return self._unit_convert(values, unit)
        else:
            return self._period_hourly_temperatures(periods, unit)

    def _period_average_temperature(self, period, unit):
        self._fetch_period(period)
        value = self.tempC[period.start:period.end - timedelta(seconds=1)].mean()
        return self._unit_convert(value, unit)

    def _period_daily_temperatures(self, period, unit):
        self._fetch_period(period)
        temps = []
        for days in range(period.timedelta.days):
            dt = period.start + timedelta(days=days)
            temps.append(self.datetime_average_temperature(dt, unit))
        return np.array(temps)

    def _period_hourly_temperatures(self, period, unit):
        self._fetch_period(period)
        temps = []
        for seconds in range(0, int(period.timedelta.total_seconds()), 3600):
            dt = period.start + timedelta(seconds=seconds)
            temps.append(self.datetime_hourly_temperature(dt, unit))
        return np.array(temps)

    def datetime_average_temperature(self, dt, unit):
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

        try:
            value = self.tempC[dt.strftime("%Y-%m-%d")].mean()
            return self._unit_convert(value, unit)
        except KeyError:
            try:
                self._fetch_datetime(dt)
                value = self.tempC[dt.strftime("%Y-%m-%d")].mean()
                return self._unit_convert(value, unit)
            except KeyError:
                pass
        return np.nan

    def datetime_hourly_temperature(self, dt, unit):
        """The hourly observed temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
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
            return self._unit_convert(self.tempC[dt], unit)
        except KeyError:
            try:
                self._fetch_datetime(dt)
                return self._unit_convert(self.tempC[dt], unit)
            except KeyError:
                pass
        return np.nan

    def hdd(self, periods, unit, base, per_day=False):
        """The total heating degree days observed during each time period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
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

        if is_list_like(periods):
            return np.array([self._period_hdd(p, unit, base, per_day) for p in periods])
        else:
            return self._period_hdd(periods, unit, base, per_day)

    def cdd(self, periods, unit, base, per_day=False):
        """The total cooling degree days observed during each time period.

        Parameters
        ----------
        periods : list of eemeter.evaluation.Period objects
            Time periods over which cooling degree days will be calculated and
            collected. A single Period may be given.
        unit : {"degC", "degF"}
            The temperature unit to be used
        base : int or float
            The base of the cooling degree day
        per_day : bool, default=False
            If True, the total should be returned as an average instead of
            a sum.

        Returns
        -------
        out : np.ndarray
            Array of cooling degree days observed during each time period.
        """

        if is_list_like(periods):
            return np.array([self._period_cdd(p, unit, base, per_day) for p in periods])
        else:
            return self._period_cdd(periods, unit, base, per_day)

    def _period_hdd(self, period, unit, base, per_day):
        """The total heating degree days observed during a particular
        time period.

        Parameters
        ----------
        period : eemeter.evaluation.Period
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
        masked_temps = np.ma.masked_array(temps, np.isnan(temps))
        total_hdd = np.sum(np.maximum(base - masked_temps,0))
        if per_day:
            n_days = period.timedelta.days
            return total_hdd / n_days
        else:
            return total_hdd

    def _period_cdd(self, period, unit, base, per_day):
        """The total cooling degree days observed during a particular
        time period.

        Parameters
        ----------
        period : eemeter.evaluation.Period
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
        masked_temps = np.ma.masked_array(temps, np.isnan(temps))
        total_cdd = np.sum(np.maximum(masked_temps - base,0))
        if per_day:
            n_days = period.timedelta.days
            return total_cdd / n_days
        else:
            return total_cdd

    def json(self):
        return {
            "station": self.station,
            "records": [{
                "datetime": d.strftime(self.cache_date_format),
                "tempC": t if pd.notnull(t) else None,
            } for d, t in self.tempC.iteritems()]
        }


class CachedWeatherSourceBase(WeatherSourceBase):

    cache_date_format = None
    cache_filename_format = None
    freq = None

    def __init__(self, station, cache_directory=None, cache_filename=None):
        super(CachedWeatherSourceBase, self).__init__(station)

        if cache_filename is None:
            self.cache_filename = self.get_cache_filename(cache_directory)
        else:
            self.cache_filename = cache_filename

        self.load_from_cache()

    def get_cache_filename(self, cache_directory=None):
        if cache_directory is None:
            cache_directory = self.get_cache_directory()
        filename = self.cache_filename_format.format(self.station)
        return os.path.join(cache_directory, filename)

    def get_cache_directory(self):
        """ Returns a directory to be used for caching.
        """
        directory = os.environ.get("EEMETER_WEATHER_CACHE_DIRECTORY", os.path.expanduser('~/.eemeter/cache'))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def save_to_cache(self):
        data = [[d.strftime(self.cache_date_format), t if pd.notnull(t) else None] for d,t in self.tempC.iteritems()]
        with open(self.cache_filename, 'w') as f:
            json.dump(data,f)

    def load_from_cache(self):
        try:
            with open(self.cache_filename, 'r') as f:
                data = json.load(f)
        except IOError:
            return
        except ValueError: # Corrupted json file
            self.clear_cache()
            return
        index = pd.to_datetime([d[0] for d in data], format=self.cache_date_format)
        values = [d[1] for d in data]

        # changed for pandas > 0.18
        self.tempC = pd.Series(values, index=index, dtype=float).sort_index().resample(self.freq).mean()

    def clear_cache(self):
        try:
            os.remove(self.cache_filename)
        except OSError:
            pass


class NOAAWeatherSourceBase(CachedWeatherSourceBase):

    year_existence_format = None
    client = NOAAClient()

    def __init__(self, station, start_year=None, end_year=None,
            cache_directory=None, cache_filename=None):
        super(NOAAWeatherSourceBase, self).__init__(station, cache_directory,
                cache_filename)

        self._year_fetches_attempted = set()

        self.station_id = station

        if start_year is not None and end_year is not None:
            self.add_year_range(start_year, end_year)
        elif start_year is not None:
            self.add_year_range(start_year, date.today().year)
        elif end_year is not None:
            self.add_year_range(date.today().year, end_year)

        self._check_for_recent_data()

    def add_year_range(self, start_year, end_year, force=False):
        """Adds temperature data to internal pandas timeseries across a
        range of years.

        Parameters
        ----------
        start_year : {int, string}
            The earliest year for which data should be fetched, e.g. "2010".
        end_year : {int, string}
            The latest year for which data should be fetched, e.g. "2013".
        force : bool, default=False
            If True, forces the fetch; if false, checks to see if year
            has been added before actually fetching.
        """
        for year in range(start_year, end_year + 1):
            self.add_year(year, force)

    def add_year(self, year, force=False):
        message = "Inheriting classes must override this method."
        raise NotImplementedError(message)

    def _year_fetch_attempted(self, year):
        return year in self._year_fetches_attempted

    def _year_in_series(self, year):
        return self.year_existence_format.format(year) in self.tempC

    def _check_for_recent_data(self):
        yesterday = date.today() - timedelta(days=1)
        if yesterday in self.tempC and pd.isnull(self.tempC[yesterday]):
            self.add_year(yesterday.year, force=True)

    def _fetch_period(self, period):
        years = []
        if period.start is not None and period.end is not None:
            self.add_year_range(period.start.year, period.end.year)
        elif period.start is not None:
            self.add_year(period.start.year)
        elif period.end is not None:
            self.add_year(period.end.year)

    def _fetch_datetime(self, dt):
        self.add_year(dt.year)


class GSODWeatherSource(NOAAWeatherSourceBase):

    cache_date_format = "%Y%m%d"
    cache_filename_format = "GSOD-{}.json"
    year_existence_format = "{}-01-01"
    freq = "D"

    def add_year(self, year, force=False):
        """Adds temperature data to internal pandas timeseries

        Parameters
        ----------
        year : {int, string}
            The year for which data should be fetched, e.g. "2010".
        force : bool, default=False
            If True, forces the fetch; if false, checks to see if year
            has been added before actually fetching.
        """

        if not force and self._year_fetch_attempted(year):
            if self._year_in_series(year):
                return
            else:
                dates = pd.date_range("{}-01-01".format(year),"{}-12-31".format(year), freq=self.freq)
                new_series = pd.Series(None, index=dates, dtype=float)
                self.tempC = self.tempC.append(new_series).sort_index().resample(self.freq).mean()
                self.save_to_cache()
                return

        data = self.client.get_gsod_data(self.station, year)
        dates = pd.date_range("{}-01-01".format(year),"{}-12-31".format(year), freq=self.freq)
        new_series = pd.Series(None, index=dates, dtype=float)
        for day in data:
            if not pd.isnull(day["temp_C"]):
                new_series[day["date"]] = day["temp_C"]

        # changed for pandas > 0.18
        self.tempC = self.tempC.append(new_series).sort_index().resample(self.freq).mean()
        self.save_to_cache()
        self._year_fetches_attempted.add(year)


class ISDWeatherSource(NOAAWeatherSourceBase):

    cache_date_format = "%Y%m%d%H"
    cache_filename_format = "ISD-{}.json"
    year_existence_format = "{}-01-01 00"
    freq = "H"

    def add_year(self, year, force=False):
        """Adds temperature data to internal pandas timeseries

        Parameters
        ----------
        year : {int, string}
            The year for which data should be fetched, e.g. "2010".
        """
        if not force and self._year_fetch_attempted(year):
            if self._year_in_series(year):
                return
            else:
                dates = pd.date_range("{}-01-01 00:00".format(year),"{}-01-01 00:00".format(int(year) + 1), freq=self.freq)[:-1]
                new_series = pd.Series(None, index=dates, dtype=float)
                self.tempC = self.tempC.append(new_series).sort_index().resample(self.freq).mean()
                self.save_to_cache()
                return

        data = self.client.get_isd_data(self.station, year)
        dates = pd.date_range("{}-01-01 00:00".format(year),"{}-01-01 00:00".format(int(year) + 1), freq=self.freq)[:-1]
        new_series = pd.Series(None, index=dates, dtype=float)
        for hour in data:
            if not pd.isnull(hour["temp_C"]):
                dt = hour["datetime"]
                new_series[datetime(dt.year, dt.month, dt.day, dt.hour)] = hour["temp_C"]

        # changed for pandas > 0.18
        self.tempC = self.tempC.append(new_series).sort_index().resample(self.freq).mean()
        self.save_to_cache()
        self._year_fetches_attempted.add(year)


class TMY3WeatherSource(CachedWeatherSourceBase):

    cache_date_format = "%Y%m%d%H"
    cache_filename_format = "TMY3-{}.json"
    freq = "H"
    client = TMY3Client()

    def __init__(self, station, station_fallback=True):
        super(TMY3WeatherSource, self).__init__(station)
        self.station_id = station
        self.station_fallback = station_fallback
        if self.tempC.shape[0] != 365 * 24:
            self._load_data()

    def _load_data(self):
        data = self.client.get_tmy3_data(self.station, self.station_fallback)
        if data is None:
            temps = [np.nan for _ in range(365 * 24)]
            index = [datetime(1900, 1, 1) + timedelta(seconds=i*3600) for i in range(365 * 24)]
        else:
            temps = [d["temp_C"] for d in data]
            index = [datetime(1900, d["dt"].month, d["dt"].day, d["dt"].hour) for d in data]

        # changed for pandas > 0.18
        self.tempC = pd.Series(temps, index=index, dtype=float).sort_index().resample('H').mean()
        self.save_to_cache()

    def annual_daily_temperatures(self, unit):
        """Returns a list of daily temperature normals for a typical
        meteorological year.

        Parameters
        ----------
        unit : {"degC", "degF"}
            The unit in which temperatures should be returned.

        Returns
        -------
        out : np.ndarray
            List with single element which is an array of observed
            daily temperatures.

        """

        periods = [Period(start=datetime(1900,1,1), end=datetime(1901,1,1))]
        return self.daily_temperatures(periods, unit)

    def _fetch_period(self, period):
        pass # loaded at init

    def _fetch_datetime(self, dt):
        pass # loaded at init

    def _normalize_period(self, period):
        start = self._normalize_datetime(period.start)
        year_offset = period.end.year - period.start.year
        end = self._normalize_datetime(period.end, year_offset)
        return Period(start, end)

    @staticmethod
    def _normalize_datetime(dt, year_offset=0):
        return datetime(1900 + year_offset, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    def _period_average_temperature(self, period, unit):
        period = self._normalize_period(period)
        return super(TMY3WeatherSource, self)._period_average_temperature(period, unit)

    def _period_daily_temperatures(self, period, unit):
        period = self._normalize_period(period)
        return super(TMY3WeatherSource, self)._period_daily_temperatures(period, unit)

    def _period_hourly_temperatures(self, period, unit):
        period = self._normalize_period(period)
        return super(TMY3WeatherSource, self)._period_hourly_temperatures(period, unit)

    def datetime_average_temperature(self, dt, unit):
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
        dt = self._normalize_datetime(dt)
        return super(TMY3WeatherSource, self).datetime_average_temperature(dt, unit)

    def datetime_hourly_temperature(self, dt, unit):
        """The hourly observed temperatures for each period.

        Parameters
        ----------
        periods : [list of] eemeter.evaluation.Period
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
        dt = self._normalize_datetime(dt)
        return super(TMY3WeatherSource, self).datetime_hourly_temperature(dt, unit)
