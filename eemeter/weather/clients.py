from eemeter.weather.location import _load_station_to_lat_lng_index, haversine

import ftplib
import gzip
from io import BytesIO
import json
import logging
from pkg_resources import resource_stream
import warnings
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class NOAAClient(object):

    def __init__(self, n_tries=3):
        self.n_tries = n_tries
        self.ftp = None  # lazily load
        self.station_index = None  # lazily load

    def _get_ftp_connection(self):
        for _ in range(self.n_tries):
            try:
                ftp = ftplib.FTP("ftp.ncdc.noaa.gov")
            except ftplib.all_errors as e:
                logger.warn("FTP connection issue: %s", e)
            else:
                try:
                    ftp.login()
                    return ftp
                except ftplib.all_errors as e:
                    logger.warn("FTP login issue: %s", e)
        raise RuntimeError("Couldn't establish an FTP connection.")

    def _load_station_index(self):
        with resource_stream('eemeter.resources',
                             'GSOD-ISD_station_index.json') as f:
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
            except (IOError, ftplib.error_perm) as e1:
                message = (
                    "Failed FTP RETR for station {}: {}"
                    .format(station_id, e1)
                )
                logger.warn(message)
            except (ftplib.error_temp, EOFError) as e2:
                # Bad connection. attempt to reconnect.
                message = (
                    "Failed FTP RETR for station {}: {}."
                    " Attempting reconnect."
                    .format(station_id, e2)
                )
                logger.warn(message)
                self.ftp.close()
                self.ftp = self._get_ftp_connection()
                try:
                    self.ftp.retrbinary('RETR {}'.format(filename),
                                        string.write)
                    break
                except (IOError, ftplib.error_perm) as e3:
                    message = (
                        "Failed FTP RETR for station {}: {}."
                        " Attempting reconnect."
                        .format(station_id, e3)
                    )
                    logger.warn(message)

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
            columns = line.split()
            date_str = columns[2].decode('utf-8')
            temp_F = float(columns[3])
            temp_C = (5./9.) * (temp_F - 32.)
            dt = datetime.strptime(date_str, "%Y%m%d").date()
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
        self.stations = None  # lazily load
        self.station_to_lat_lng = None  # lazily load

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
                message = (
                    "Using station {} instead.".format(nearby_station)
                )
                warnings.warn(message)
                return nearby_station
        return None

    def get_tmy3_data(self, station, station_fallback=True):

        if self.stations is None:
            self.stations = self._load_stations()

        if station not in self.stations:
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

        url = (
            "http://rredc.nrel.gov/solar/old_data/nsrdb/"
            "1991-2005/data/tmy3/{}TYA.CSV".format(station)
        )
        r = requests.get(url)

        if r.status_code == 200:
            hours = []
            for line in r.text.splitlines()[2:]:
                row = line.split(",")
                year = row[0][6:10]
                month = row[0][0:2]
                day = row[0][3:5]
                hour = int(row[1][0:2]) - 1

                # YYYYMMDDHH
                date_string = "{}{}{}{:02d}".format(year, month, day, hour)

                dt = datetime.strptime(date_string, "%Y%m%d%H")
                temp_C = float(row[31])

                hours.append({"temp_C": temp_C, "dt": dt})
            return hours
        else:
            message = (
                "Station {} was not found. Tried url {}.".format(station, url)
            )
            warnings.warn(message)
            return None
