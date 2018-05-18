from datetime import datetime, date
from pkg_resources import resource_stream

import json
import pandas as pd
import pytz

class WeatherSource(object):

    def __init__(self, station, normalized, use_cz2010):
        self.station_index = None
        self.resource_loc = None
        self.normalized = normalized
        self.use_cz2010 = use_cz2010
        self.station = station
        self.tempC = pd.Series(dtype=float)
        self._assign_station_type_and_loc(normalized, use_cz2010)
        self._check_station(station)

    def _assign_station_type_and_loc(self, normalized, use_cz2010):
        if normalized:
            if use_cz2010:
                self.resource_loc = 'supported_cz2010_stations.json'
                self.station_type = 'CZ2010'
            else:
                self.resource_loc = 'supported_tmy3_stations.json'
                self.station_type = 'TMY3'
        else:
            self.resource_loc ='GSOD-ISD_station_index.json'
            self.station_type = 'ISD'

    def _check_station(self, station):
        index = self._load_station_index()
        if station not in index:
            message = (
                "`{}` not recognized as valid USAF weather station identifier."
            )
            raise ValueError(message)

    def _load_station_index(self):

        if self.station_index is None:
            with resource_stream('eemeter.resources',self.resource_loc) as f:
                self.station_index = json.loads(f.read().decode("utf-8"))
        return self.station_index

    def __repr__(self):
        return '{}WeatherSource("{}")'.format(self.station_type, self.station)
