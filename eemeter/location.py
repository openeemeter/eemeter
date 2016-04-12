import json
import numpy as np
from pkg_resources import resource_stream

station_to_lat_lng_index = None
station_to_zipcodes_index = None
station_to_climate_zone_index = None

zipcode_to_lat_lng_index = None
zipcode_to_station_index = None
zipcode_to_climate_zone_index = None

climate_zone_to_stations_index = None
climate_zone_to_zipcodes_index = None

def _get_json_resource(filename):
    with resource_stream('eemeter.resources', filename) as f:
        resource = json.loads(f.read().decode('utf-8'))
    return resource


def _load_station_to_lat_lng_index():
    global station_to_lat_lng_index
    if station_to_lat_lng_index is None:
        station_to_lat_lng_index = _get_json_resource('usaf_station_lat_long.json')
    return station_to_lat_lng_index

def _load_station_to_zipcodes_index():
    global station_to_zipcodes_index
    if station_to_zipcodes_index is None:
        station_to_zipcodes_index = _get_json_resource('usaf_station_zipcodes.json')
    return station_to_zipcodes_index

def _load_station_to_climate_zone_index():
    global station_to_climate_zone_index
    if station_to_climate_zone_index is None:
        station_to_climate_zone_index = _get_json_resource('usaf_station_climate_zone.json')
    return station_to_climate_zone_index


def _load_zipcode_to_lat_lng_index():
    global zipcode_to_lat_lng_index
    if zipcode_to_lat_lng_index is None:
        zipcode_to_lat_lng_index = _get_json_resource('zipcode_centroid_lat_long.json')
    return zipcode_to_lat_lng_index

def _load_zipcode_to_station_index():
    global zipcode_to_station_index
    if zipcode_to_station_index is None:
        zipcode_to_station_index = _get_json_resource('zipcode_usaf_station.json')
    return zipcode_to_station_index

def _load_zipcode_to_climate_zone_index():
    global zipcode_to_climate_zone_index
    if zipcode_to_climate_zone_index is None:
        zipcode_to_climate_zone_index = _get_json_resource('zipcode_climate_zone.json')
    return zipcode_to_climate_zone_index


def _load_climate_zone_to_zipcodes_index():
    global climate_zone_to_zipcodes_index
    if climate_zone_to_zipcodes_index is None:
        climate_zone_to_zipcodes_index = _get_json_resource('climate_zone_zipcodes.json')
    return climate_zone_to_zipcodes_index

def _load_climate_zone_to_stations_index():
    global climate_zone_to_stations_index
    if climate_zone_to_stations_index is None:
        climate_zone_to_stations_index = _get_json_resource('climate_zone_usaf_stations.json')
    return climate_zone_to_stations_index

def haversine(lat1, lng1, lat2, lng2):
    """ Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Parameters
    ----------
    lat1 : float
        Latitude coordinate of first point.
    lng1 : float
        Longitude coordinate of first point.
    lat2 : float
        Latitude coordinate of second point.
    lng2 : float
        Longitude coordinate of second point.

    Returns
    -------
    distance : float
        Kilometers between the two lat/lng coordinates.
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

def lat_lng_to_station(lat, lng):
    """Return the closest USAF station ID using latitude and
    longitude coordinates.

    Parameters
    ----------
    lat : float
        Latitude coordinate.
    lng : float
        Longitude coordinate.

    Returns
    -------
    station : str, None
        String representing a USAF weather station ID or None, if none was found.
    """
    if lat is None or lng is None:
        return None
    station_to_lat_lng_index = _load_station_to_lat_lng_index()
    index_list = list(station_to_lat_lng_index.items())
    dists = [haversine(lat, lng, stat_lat, stat_lng)
            for _, (stat_lat, stat_lng) in index_list]
    return index_list[np.argmin(dists)][0]

def lat_lng_to_zipcode(lat, lng):
    """Return the closest ZIP code using latitude and
    longitude coordinates.

    Parameters
    ----------
    lat : float
        Latitude coordinate.
    lng : float
        Longitude coordinate.

    Returns
    -------
    zipcode : str, None
        String representing a USPS ZIP code, or None, if none was found.

    """

    if lat is None or lng is None:
        return None
    zipcode_to_lat_lng_index = _load_zipcode_to_lat_lng_index()
    index_list = list(zipcode_to_lat_lng_index.items())
    dists = [haversine(lat, lng, zip_lat, zip_lng)
            for _, (zip_lat, zip_lng) in index_list]
    return index_list[np.argmin(dists)][0]

def station_to_lat_lng(station):
    """Return the latitude and longitude coordinates of the given station.

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    lat_lng : tuple of float
        Latitude and longitude coordinates.

    """
    lat_lng = _load_station_to_lat_lng_index().get(station)
    if lat_lng is None:
        return None, None
    return lat_lng

def station_to_zipcodes(station):
    """Return the zipcodes that map to this station.

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    zipcode : list of str
        String representing a USPS ZIP code.

    """
    return _load_station_to_zipcodes_index().get(station)

def station_to_climate_zone(station):
    """Return the climate_zone of the station.

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    climate_zone : str
        String representing a USPS ZIP code.

    """
    return _load_station_to_climate_zone_index().get(station)

def zipcode_to_lat_lng(zipcode):
    """Return the latitude and longitude centroid of a particular ZIP code.

    Parameters
    ----------
    zipcode : str
        String representing a USPS ZIP code.

    Returns
    -------
    lat_lng : tuple of float
        Latitude and longitude coordinates.

    """
    lat_lng = _load_zipcode_to_lat_lng_index().get(zipcode)
    if lat_lng is None:
        return None, None
    return lat_lng

def zipcode_to_station(zipcode):
    """Return the nearest USAF station (by latitude and longitude centroid) of
    the ZIP code.

    Parameters
    ----------
    zipcode : str
        String representing a USPS ZIP code.

    Returns
    -------
    station : str
        String representing a USAF Weather station ID
    """
    return _load_zipcode_to_station_index().get(zipcode)

def zipcode_to_climate_zone(zipcode):
    """Return the climate zone of the ZIP code (by latitude and longitude
    centroid of ZIP code).

    Parameters
    ----------
    zipcode : str
        String representing a USPS ZIP code.

    Returns
    -------
    climate_zone : str
        String representing a climate zone
    """
    return _load_zipcode_to_climate_zone_index().get(zipcode)

def climate_zone_to_zipcodes(climate_zone):
    """Return ZIP codes with centroids in the given climate zone.

    Parameters
    ----------
    climate_zone : str
        String representing a climate zone.

    Returns
    -------
    zipcodes : list of str
        Strings representing USPS ZIP codes.
    """
    return _load_climate_zone_to_zipcodes_index().get(climate_zone)

def climate_zone_to_stations(climate_zone):
    """Return weather stations falling within in the given climate zone.

    Parameters
    ----------
    climate_zone : str
        String representing a climate zone.

    Returns
    -------
    stations : list of str
        Strings representing USAF station ids.
    """
    return _load_climate_zone_to_stations_index().get(climate_zone)

class Location(object):
    """ Represents a project location. Should be initialized with one of
    `lat_lng`, `zipcode`, or `station`

    Parameters
    ----------
    lat_lng : tuple of float
        Latitude and longitude coordinates.
    zipcode : str
        String representing a USPS ZIP code; e.g. "60642"
    station : str
        String representing a USAF Weather station ID
    """

    def __init__(self, lat_lng=None, zipcode=None, station=None):
        if lat_lng is not None:
            self.lat, self.lng = lat_lng
            self.zipcode = lat_lng_to_zipcode(self.lat, self.lng)
            self.station = lat_lng_to_station(self.lat, self.lng)
        elif zipcode is not None:
            self.lat, self.lng = zipcode_to_lat_lng(zipcode)
            self.zipcode = zipcode
            self.station = zipcode_to_station(zipcode)
        elif station is not None:
            self.lat, self.lng = station_to_lat_lng(station)
            self.zipcode = lat_lng_to_zipcode(self.lat, self.lng)
            self.station = station
        else:
            message = "Please supply a lat/long, ZIP code or Weather Station ID"
            raise ValueError(message)
