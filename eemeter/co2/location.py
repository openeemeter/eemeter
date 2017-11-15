import json
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


def _load_zipcode_to_avert_region():
    return _load_resource('zipcode_to_avert_region',
                          'zipcode_avert_region.json')


def _load_supported_zipcodes_index():
    return _load_resource('supported_zipcodes_index',
                          'supported_zipcodes.json')


def zipcode_to_avert_region(zipcode):
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
    region = _load_zipcode_to_avert_region().get(zipcode, None)
    if region is None:
        return None
    return region
