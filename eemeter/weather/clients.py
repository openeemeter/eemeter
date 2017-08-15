import ftplib
import gzip
from io import BytesIO
import json
import logging
from pkg_resources import resource_stream
import warnings
from datetime import datetime, timedelta

import pytz
import pandas as pd
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
                logger.info(
                    "Successfully established connection to ftp.ncdc.noaa.gov."
                )
                try:
                    ftp.login()
                except ftplib.all_errors as e:
                    logger.warn("FTP login issue: %s", e)
                else:
                    logger.info(
                        "Successfully logged in to ftp.ncdc.noaa.gov."
                    )
                    return ftp
        raise RuntimeError("Couldn't establish an FTP connection.")

    def _load_station_index(self):
        if self.station_index is None:
            with resource_stream('eemeter.resources',
                                 'GSOD-ISD_station_index.json') as f:
                self.station_index = json.loads(f.read().decode("utf-8"))
        return self.station_index

    def _get_potential_station_ids(self, station):
        self._load_station_index()

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
            except (IOError, ftplib.error_perm) as e1:
                logger.warn(
                    "Failed FTP RETR for station {}: {}."
                    " Not attempting reconnect."
                    .format(station_id, e1)
                )
            except (ftplib.error_temp, EOFError) as e2:
                # Bad connection. attempt to reconnect.
                logger.warn(
                    "Failed FTP RETR for station {}: {}."
                    " Attempting reconnect."
                    .format(station_id, e2)
                )
                self.ftp.close()
                self.ftp = self._get_ftp_connection()
                try:
                    self.ftp.retrbinary('RETR {}'.format(filename),
                                        string.write)
                except (IOError, ftplib.error_perm) as e3:
                    logger.warn(
                        "Failed FTP RETR for station {}: {}."
                        " Trying another station id."
                        .format(station_id, e3)
                    )
                else:
                    break
            else:
                break

        logger.info(
            'Successfully retrieved ftp://ftp.ncdc.noaa.gov{}'
            .format(filename)
        )

        string.seek(0)
        f = gzip.GzipFile(fileobj=string)
        lines = f.readlines()
        string.close()
        return lines

    def get_gsod_data(self, station, year):

        filename_format = '/pub/data/gsod/{year}/{station}-{year}.op.gz'
        lines = self._retreive_file_lines(filename_format, station, year)

        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 00:00".format(year),
                              freq='D', tz=pytz.UTC)
        series = pd.Series(None, index=dates, dtype=float)

        for line in lines[1:]:
            columns = line.split()
            date_str = columns[2].decode('utf-8')
            temp_F = float(columns[3])
            temp_C = (5. / 9.) * (temp_F - 32.)
            dt = pytz.UTC.localize(datetime.strptime(date_str, "%Y%m%d"))
            series[dt] = temp_C

        return series

    def get_isd_data(self, station, year):

        filename_format = '/pub/data/noaa/{year}/{station}-{year}.gz'
        lines = self._retreive_file_lines(filename_format, station, year)

        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 23:00".format(int(year) + 1),
                              freq='H', tz=pytz.UTC)
        series = pd.Series(None, index=dates, dtype=float)

        for line in lines:
            if line[87:92].decode('utf-8') == "+9999":
                temp_C = float("nan")
            else:
                temp_C = float(line[87:92]) / 10.
            date_str = line[15:27].decode('utf-8')

            # there can be multiple readings per hour, so set all to minute 0
            dt = pytz.UTC.localize(datetime.strptime(date_str, "%Y%m%d%H%M")).replace(minute=0)

            # only set the temp if it's the first encountered in the hour.
            if pd.isnull(series.ix[dt]):
                series[dt] = temp_C

        return series


class TMY3Client(object):

    def __init__(self):
        self.station_index = None  # lazily load

    def _load_station_index(self):
        if self.station_index is None:
            with resource_stream('eemeter.resources',
                                 'supported_tmy3_stations.json') as f:
                self.station_index = set(json.loads(f.read().decode("utf-8")))
        return self.station_index

    def get_hourly_weather_normal_data(self, station):

        self._load_station_index()

        if station not in self.station_index:
            message = (
                "Station {} is not a TMY3 station."
                " See eemeter/resources/supported_tmy3_stations.json for a"
                " complete list of stations."
                .format(station)
            )
            raise ValueError(message)

        url = (
            "http://rredc.nrel.gov/solar/old_data/nsrdb/"
            "1991-2005/data/tmy3/{}TYA.CSV".format(station)
        )
        r = requests.get(url)

        index = pd.date_range("1900-01-01 00:00", "1900-12-31 23:00",
                              freq='H', tz=pytz.UTC)
        series = pd.Series(None, index=index, dtype=float)

        if r.status_code == 200:

            lines = r.text.splitlines()

            utc_offset_str = lines[0].split(',')[3]
            utc_offset = timedelta(seconds=3600 * float(utc_offset_str))

            for line in lines[2:]:
                row = line.split(",")
                month = row[0][0:2]
                day = row[0][3:5]
                hour = int(row[1][0:2]) - 1

                # YYYYMMDDHH
                date_string = "1900{}{}{:02d}".format(month, day, hour)

                dt = datetime.strptime(date_string, "%Y%m%d%H") - utc_offset

                # Only a little redundant to make year 1900 again - matters for
                # first or last few hours of the year depending UTC on offset
                dt = pytz.UTC.localize(dt.replace(year=1900))
                temp_C = float(row[31])

                series[dt] = temp_C

        else:
            message = (
                "Station {} was not found. Tried url {}.".format(station, url)
            )
            warnings.warn(message)

        return series


class CZ2010Client(object):

    def __init__(self):
        self.station_index = None

    def _load_station_index(self):
        if self.station_index is None:
            with resource_stream('eemeter.resources',
                                 'supported_cz2010_stations.json') as f:
                self.station_index = set(json.loads(f.read().decode("utf-8")))
        return self.station_index

    def get_hourly_weather_normal_data(self, station):

        # Note: at time of writing, this loading code is exactly the same
        # as the TMY3 loading code. No detectable difference in the format
        # for these purposes (i.e., getting temperature). The only difference
        # is the URL from which the data is pulled.

        self._load_station_index()

        if station not in self.station_index:
            message = (
                "Station {} is not a CZ2010 station."
                " See eemeter/resources/supported_cz2010_stations.json for a"
                " complete list of stations."
                .format(station)
            )
            raise ValueError(message)

        # NOTE: This URL is hardcoded but the data may not always be available
        # from this source. Set with env variable instead?
        url = (
            "https://storage.googleapis.com/oee-cz2010/csv/{}_CZ2010.CSV"
            .format(station)
        )
        r = requests.get(url)

        index = pd.date_range("1900-01-01 00:00", "1900-12-31 23:00",
                              freq='H', tz=pytz.UTC)
        series = pd.Series(None, index=index, dtype=float)

        if r.status_code == 200:

            lines = r.text.splitlines()

            utc_offset_str = lines[0].split(',')[3]
            utc_offset = timedelta(seconds=3600 * float(utc_offset_str))

            for line in lines[2:]:
                row = line.split(",")
                month = row[0][0:2]
                day = row[0][3:5]
                hour = int(row[1][0:2]) - 1

                # YYYYMMDDHH
                date_string = "1900{}{}{:02d}".format(month, day, hour)

                dt = datetime.strptime(date_string, "%Y%m%d%H") - utc_offset

                # Only a little redundant to make year 1900 again - matters for
                # first or last few hours of the year depending UTC on offset
                dt = pytz.UTC.localize(dt.replace(year=1900))
                temp_C = float(row[31])

                series[dt] = temp_C

        else:
            message = (
                "Station {} was not found. Tried url {}.".format(station, url)
            )
            warnings.warn(message)

        return series
