import json
import numpy as np
from pkg_resources import resource_stream

tmy3_to_lat_lng_index = None
tmy3_to_zipcode_index = None
zipcode_to_lat_lng_index = None
zipcode_to_tmy3_index = None

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
