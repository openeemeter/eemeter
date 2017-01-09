import json
import numpy as np
from pkg_resources import resource_stream


resources = {}


def _get_json_resource(filename):
    with resource_stream('eemeter.resources', filename) as f:
        resource = json.loads(f.read().decode('utf-8'))
    return resource


def _load_resource(name, filename):
    global resources
    if resources.get(name, None) is None:
        resources[name] = _get_json_resource(filename)
    return resources[name]


def _load_usaf_station_to_lat_lng_index():
    return _load_resource('usaf_station_to_lat_lng_index',
                          'usaf_station_lat_lngs.json')


def _load_usaf_station_to_zipcodes_index():
    return _load_resource('usaf_station_to_zipcodes_index',
                          'usaf_station_zipcodes.json')


def _load_usaf_station_to_climate_zone_index():
    return _load_resource('usaf_station_to_climate_zone_index',
                          'usaf_station_climate_zone.json')


def _load_tmy3_station_to_lat_lng_index():
    return _load_resource('tmy3_station_to_lat_lng_index',
                          'tmy3_station_lat_lngs.json')


def _load_tmy3_station_to_zipcodes_index():
    return _load_resource('tmy3_station_to_zipcodes_index',
                          'tmy3_station_zipcodes.json')


def _load_tmy3_station_to_climate_zone_index():
    return _load_resource('tmy3_station_to_climate_zone_index',
                          'tmy3_station_climate_zone.json')


def _load_zipcode_to_lat_lng_index():
    return _load_resource('zipcode_to_lat_lng_index',
                          'zipcode_centroid_lat_lngs.json')


def _load_zipcode_to_usaf_station_index():
    return _load_resource('zipcode_to_usaf_station_index',
                          'zipcode_usaf_station.json')


def _load_zipcode_to_tmy3_station_index():
    return _load_resource('zipcode_to_tmy3_station_index',
                          'zipcode_tmy3_station.json')


def _load_zipcode_to_climate_zone_index():
    return _load_resource('zipcode_to_climate_zone_index',
                          'zipcode_climate_zone.json')


def _load_climate_zone_to_zipcodes_index():
    return _load_resource('climate_zone_to_zipcodes_index',
                          'climate_zone_zipcodes.json')


def _load_climate_zone_to_usaf_stations_index():
    return _load_resource('climate_zone_to_usaf_stations_index',
                          'climate_zone_usaf_stations.json')


def _load_climate_zone_to_tmy3_stations_index():
    return _load_resource('climate_zone_to_tmy3_stations_index',
                          'climate_zone_tmy3_stations.json')


def _load_supported_zipcodes_index():
    return _load_resource('supported_zipcodes_index',
                          'supported_zipcodes.json')


def _load_supported_tmy3_stations_index():
    return _load_resource('supported_tmy3_stations_index',
                          'supported_tmy3_stations.json')


def _load_supported_usaf_stations_index():
    return _load_resource('supported_usaf_stations_index',
                          'supported_usaf_stations.json')


def _load_supported_climate_zones_index():
    return _load_resource('supported_climate_zones_index',
                          'supported_climate_zones.json')


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
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3959 for miles
    return c * r


def lat_lng_to_usaf_station(lat, lng):
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
        String representing a USAF weather station ID or None, if none was
        found.
    """
    if lat is None or lng is None:
        return None
    usaf_station_to_lat_lng_index = _load_usaf_station_to_lat_lng_index()
    index_list = list(usaf_station_to_lat_lng_index.items())
    dists = [haversine(lat, lng, stat_lat, stat_lng)
             for _, (stat_lat, stat_lng) in index_list]
    return index_list[np.argmin(dists)][0]


def lat_lng_to_tmy3_station(lat, lng):
    """Return the closest TMY3 station ID using latitude and
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
        String representing a TMY3 weather station ID or None, if none was
        found.
    """
    if lat is None or lng is None:
        return None
    tmy3_station_to_lat_lng_index = _load_tmy3_station_to_lat_lng_index()
    index_list = list(tmy3_station_to_lat_lng_index.items())
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


def lat_lng_to_climate_zone(lat, lng):
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
    climate_zone : str, None
        String representing a climate zone.

    """
    zipcode = lat_lng_to_zipcode(lat, lng)
    zipcode_to_climate_zone_index = _load_zipcode_to_climate_zone_index()
    return zipcode_to_climate_zone_index.get(zipcode, None)


def usaf_station_to_lat_lng(station):
    """Return the latitude and longitude coordinates of the given USAF station.

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    lat_lng : tuple of float
        Latitude and longitude coordinates.

    """
    lat_lng = _load_usaf_station_to_lat_lng_index().get(station, None)
    if lat_lng is None:
        return None, None
    return lat_lng


def usaf_station_to_zipcodes(station):
    """Return the zipcodes that map to this USAF station.

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    zipcodes : list of str
        Strings representing a USPS ZIP code mapped to from this station.

    """
    return _load_usaf_station_to_zipcodes_index().get(station, None)


def usaf_station_to_climate_zone(station):
    """Return the climate zone of the station.

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    climate_zone : str
        String representing a climate zone

    """
    return _load_usaf_station_to_climate_zone_index().get(station, None)


def tmy3_station_to_lat_lng(station):
    """Return the latitude and longitude coordinates of the given station.

    Parameters
    ----------
    station : str
        String representing a TMY3 USAF Weather station ID

    Returns
    -------
    lat_lng : tuple of float
        Latitude and longitude coordinates.

    """
    lat_lng = _load_tmy3_station_to_lat_lng_index().get(station, None)
    if lat_lng is None:
        return None, None
    return lat_lng


def tmy3_station_to_zipcodes(station):
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
    return _load_tmy3_station_to_zipcodes_index().get(station, None)


def tmy3_station_to_climate_zone(station):
    """Return the climate zone of the station.

    Parameters
    ----------
    station : str
        String representing a USAF Weather station ID

    Returns
    -------
    climate_zone : str
        String representing a climate zone.

    """
    return _load_tmy3_station_to_climate_zone_index().get(station, None)


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
    lat_lng = _load_zipcode_to_lat_lng_index().get(zipcode, None)
    if lat_lng is None:
        return None, None
    return lat_lng


def zipcode_to_usaf_station(zipcode):
    """Return the nearest USAF station (by latitude and longitude centroid) of
    the ZIP code.

    Parameters
    ----------
    zipcode : str
        String representing a USPS ZIP code.

    Returns
    -------
    station : str
        String representing a USAF weather station ID
    """
    return _load_zipcode_to_usaf_station_index().get(zipcode, None)


def zipcode_to_tmy3_station(zipcode):
    """Return the nearest TMY3 station (by latitude and longitude centroid) of
    the ZIP code.

    Parameters
    ----------
    zipcode : str
        String representing a USPS ZIP code.

    Returns
    -------
    station : str
        String representing a TMY3 Weather station (USAF ID).
    """
    return _load_zipcode_to_tmy3_station_index().get(zipcode, None)


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
    return _load_zipcode_to_climate_zone_index().get(zipcode, None)


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
    return _load_climate_zone_to_zipcodes_index().get(climate_zone, None)


def climate_zone_to_usaf_stations(climate_zone):
    """Return USAF weather stations falling within in the given climate zone.

    Parameters
    ----------
    climate_zone : str
        String representing a climate zone.

    Returns
    -------
    stations : list of str
        Strings representing USAF station ids.
    """
    return _load_climate_zone_to_usaf_stations_index().get(climate_zone, None)


def climate_zone_to_tmy3_stations(climate_zone):
    """Return TMY3 weather stations falling within in the given climate zone.

    Parameters
    ----------
    climate_zone : str
        String representing a climate zone.

    Returns
    -------
    stations : list of str
        Strings representing TMY3 station ids.
    """
    return _load_climate_zone_to_tmy3_stations_index().get(climate_zone, None)


def zipcode_is_supported(zipcode):
    """True if given ZIP Code is supported. ZCTA only.

    Parameters
    ----------
    zipcode : str
        5-digit string representing a zipcode.

    Returns
    -------
    supported : bool
        `True` if supported, otherwise `False`.
    """
    return zipcode in _load_supported_zipcodes_index()


def usaf_station_is_supported(station):
    """True if given USAF weather station is supported. USAF IDs.

    Parameters
    ----------
    station : str
        6-digit string representing a weather station.

    Returns
    -------
    supported : bool
        `True` if supported, otherwise `False`.
    """
    return station in _load_supported_usaf_stations_index()


def tmy3_station_is_supported(station):
    """True if given TMY3 weather station is supported. USAF IDs.

    Parameters
    ----------
    station : str
        6-digit string representing a weather station.

    Returns
    -------
    supported : bool
        `True` if supported, otherwise `False`.
    """
    return station in _load_supported_tmy3_stations_index()


def climate_zone_is_supported(climate_zone):
    """True if given Climate Zone is supported.

    Parameters
    ----------
    climate_zone : str
        String representing a climate_zone.

    Returns
    -------
    supported : bool
        `True` if supported, otherwise `False`.
    """
    return climate_zone in _load_supported_climate_zones_index()
