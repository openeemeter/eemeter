import json
import numpy as np
from pkg_resources import resource_stream

station_to_lat_lng_index = None
station_to_zipcode_index = None
zipcode_to_lat_lng_index = None
zipcode_to_station_index = None

def haversine(lat1,lng1,lat2,lng2):
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

def lat_lng_to_station(lat,lng):
    """Return the closest TMY3 weather station id (USAF) using latitude and
    longitude coordinates.

    Parameters
    ----------
    lat : float
        Latitude coordinate.
    lng : float
        Longitude coordinate.

    Returns
    -------
    station : str
        String representing a USAF Weather station ID
    """
    global station_to_lat_lng_index
    if station_to_lat_lng_index is None:
        with resource_stream('eemeter.resources','tmy3_to_lat_lng.json') as f:
            station_to_lat_lng_index = json.loads(f.read().decode("utf-8"))
    dists = []
    index_list = [i for i in station_to_lat_lng_index.items()]
    for station,(stat_lat,stat_lng) in index_list:
        dists.append(haversine(lat,lng,stat_lat,stat_lng))
    return index_list[np.argmin(dists)][0]

def lat_lng_to_zipcode(lat,lng):
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
    zipcode : str
        String representing a USPS ZIP code; e.g. "60642"

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
    global station_to_lat_lng_index
    if station_to_lat_lng_index is None:
        with resource_stream('eemeter.resources','tmy3_to_lat_lng.json') as f:
            station_to_lat_lng_index = json.loads(f.read().decode("utf-8"))
    return station_to_lat_lng_index.get(station)

def station_to_zipcode(station):
    """Return the nearest zipcode to the station by latitude and longitude
    centroid. (Note: Not always the same as finding the containing ZIP code
    area)

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    zipcode : str
        String representing a USPS ZIP code; e.g. "60642"

    """
    global station_to_zipcode_index
    if station_to_zipcode_index is None:
        with resource_stream('eemeter.resources','tmy3_to_zipcode.json') as f:
            station_to_zipcode_index = json.loads(f.read().decode("utf-8"))
    return station_to_zipcode_index.get(station)

def zipcode_to_lat_lng(zipcode):
    """Return the latitude and longitude centroid of a particular ZIP code.

    Parameters
    ----------
    zipcode : str
        String representing a USPS ZIP code; e.g. "60642"

    Returns
    -------
    lat_lng : tuple of float
        Latitude and longitude coordinates.

    """
    global zipcode_to_lat_lng_index
    if zipcode_to_lat_lng_index is None:
        with resource_stream('eemeter.resources','zipcode_to_lat_lng.json') as f:
            zipcode_to_lat_lng_index = json.loads(f.read().decode("utf-8"))
    return zipcode_to_lat_lng_index.get(zipcode)

def zipcode_to_station(zipcode):
    """Return the nearest TMY3 station (by latitude and longitude centroid) of
    the ZIP code.

    Parameters
    ----------
    zipcode : str
        String representing a USPS ZIP code; e.g. "60642"

    Returns
    -------
    station : str
        String representing a USAF Weather station ID
    """
    global zipcode_to_station_index
    if zipcode_to_station_index is None:
        with resource_stream('eemeter.resources','zipcode_to_tmy3.json') as f:
            zipcode_to_station_index = json.loads(f.read().decode("utf-8"))
    return zipcode_to_station_index.get(zipcode)

class Location(object):
    """ Represents a project location. SHould be initialized with one of
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

    def __init__(self, lat_lng=None, zipcode=None, station=None,
            station_finding_method="closest"):
        if lat_lng is not None:
            self.lat, self.lng = lat_lng
            self.zipcode = lat_lng_to_zipcode(self.lat,self.lng)
            self.station = lat_lng_to_station(self.lat,self.lng)
        elif zipcode is not None:
            self.lat, self.lng = zipcode_to_lat_lng(zipcode)
            self.zipcode = zipcode
            self.station = lat_lng_to_station(self.lat,self.lng)
        elif station is not None:
            self.lat, self.lng = station_to_lat_lng(station)
            self.zipcode = lat_lng_to_zipcode(self.lat,self.lng)
            self.station = station
        else:
            raise ValueError("Please supply a lat/long, ZIP code or Weather Station ID")
