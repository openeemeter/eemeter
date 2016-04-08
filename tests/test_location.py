from eemeter.location import climate_zone_to_stations
from eemeter.location import climate_zone_to_zipcodes

from eemeter.location import lat_lng_to_station
from eemeter.location import lat_lng_to_zipcode

from eemeter.location import station_to_climate_zone
from eemeter.location import station_to_lat_lng
from eemeter.location import station_to_zipcodes

from eemeter.location import zipcode_to_climate_zone
from eemeter.location import zipcode_to_lat_lng
from eemeter.location import zipcode_to_station

from eemeter.location import haversine
from eemeter.location import Location

from numpy.testing import assert_allclose

import pytest

RTOL = 1e-1
ATOL = 1e-1

@pytest.fixture(params=[
    (0,0,0,0,0),
    (76,1,76,1,0),
    (76,1,76,361,0),
    (0,0,0,90,10007.54339801),
    (0,0,0,180,20015.08679602),
    (0,-180,0,180,0),
    (-90,0,90,0,20015.08679602),
    (-90,0,90,180,20015.08679602),
    ])
def lat_lng_dist(request):
    return request.param

@pytest.fixture(params=[
    ('CA_01', '725945', 3),
    ('2_A_Hot-Humid', '722123', 150),
    ('7_NA_Very Cold', '726544', 138),
    ])
def climate_zone_station(request):
    return request.param

@pytest.fixture(params=[
    ('CA_16', '93550', 226),
    ('4_B_Mixed-Dry', '79237', 291),
    ])
def climate_zone_zipcode(request):
    return request.param

@pytest.fixture(params=[
    ((46.7, -92.1), "726427"),
    ((41.2, -96.0), "720308"),
    ((27.8, -97.4), "722510"),
    ])
def lat_lng_station(request):
    return request.param

@pytest.fixture(params=[
    ((47.6, -122.2), "98004"),
    ((33.9, -117.9), "92835"),
    ((43.4, -110.7), "83014"),
    ])
def lat_lng_zipcode(request):
    return request.param

@pytest.fixture(params=[
    ('725030', '4_A_Mixed-Humid'),
    ])
def station_climate_zone(request):
    return request.param

@pytest.fixture(params=[
    ('724345', (38.6, -90.6)),
    ])
def station_lat_lng(request):
    return request.param

@pytest.fixture(params=[
    ('725030', '11370', 96),
    ])
def station_zipcode(request):
    return request.param

@pytest.fixture(params=[
    ('88312', '4_B_Mixed-Dry'),
    ])
def zipcode_climate_zone(request):
    return request.param

@pytest.fixture(params=[
    ('88312', (33.4,-105.6)),
    ('30310', (33.7, -84.4)),
    ('15210', (40.4, -80.0)),
    ])
def zipcode_lat_lng(request):
    return request.param

@pytest.fixture(params=[
    ('33145', '722020'),
    ])
def zipcode_station(request):
    return request.param

@pytest.fixture(params=[
    ((41.28, -72.88), '06512', '997284'),
    ])
def lat_lng_zipcode_station(request):
    return request.param

##### TESTS #####

def test_haversine(lat_lng_dist):
    lat1, lng1, lat2, lng2, dist = lat_lng_dist
    haversine_dist = haversine(lat1, lng1, lat2, lng2)
    assert_allclose(haversine_dist, dist, rtol=RTOL, atol=ATOL)


def test_climate_zone_to_stations(climate_zone_station):
    climate_zone, station, n = climate_zone_station
    stations = climate_zone_to_stations(climate_zone)
    assert station in stations
    assert len(stations) == n

def test_climate_zone_to_zipcodes(climate_zone_zipcode):
    climate_zone, zipcode, n = climate_zone_zipcode
    zipcodes = climate_zone_to_zipcodes(climate_zone)
    assert zipcode in zipcodes
    assert len(zipcodes) == n


def test_lat_lng_to_station(lat_lng_station):
    lat_lng, station = lat_lng_station
    assert station == lat_lng_to_station(*lat_lng)

def test_lat_lng_to_zipcode(lat_lng_zipcode):
    lat_lng, zipcode = lat_lng_zipcode
    assert zipcode == lat_lng_to_zipcode(*lat_lng)


def test_station_to_climate_zone(station_climate_zone):
    station, climate_zone = station_climate_zone
    assert climate_zone == station_to_climate_zone(station)

def test_station_to_lat_lng(station_lat_lng):
    station, lat_lng = station_lat_lng
    s_lat_lng = station_to_lat_lng(station)
    assert_allclose(s_lat_lng, lat_lng, rtol=RTOL, atol=ATOL)

def test_station_to_lat_lng_null():
    lat, lng = station_to_lat_lng("NotValid")
    assert lat is None and lng is None

def test_station_to_zipcodes(station_zipcode):
    station, zipcode, n = station_zipcode
    zipcodes = station_to_zipcodes(station)
    assert zipcode in zipcodes
    assert len(zipcodes) == n


def test_zipcode_to_climate_zone(zipcode_climate_zone):
    zipcode, climate_zone = zipcode_climate_zone
    assert climate_zone == zipcode_to_climate_zone(zipcode)

def test_zipcode_to_lat_lng(zipcode_lat_lng):
    zipcode, lat_lng = zipcode_lat_lng
    z_lat_lng = zipcode_to_lat_lng(zipcode)
    assert_allclose(z_lat_lng, lat_lng, rtol=RTOL, atol=ATOL)

def test_zipcode_to_lat_lng_null():
    lat, lng = zipcode_to_lat_lng("NotValid")
    assert lat is None and lng is None

def test_zipcode_to_station(zipcode_station):
    zipcode, station = zipcode_station
    assert station == zipcode_to_station(zipcode)


def test_location_init(lat_lng_zipcode_station):
    lat_lng, zipcode, station = lat_lng_zipcode_station

    l = Location(station=station)
    assert_allclose(lat_lng, (l.lat, l.lng), rtol=RTOL, atol=ATOL)
    assert station == l.station
    assert zipcode == l.zipcode

    l = Location(lat_lng=lat_lng)
    assert_allclose(lat_lng, (l.lat, l.lng), rtol=RTOL, atol=ATOL)
    assert station == l.station
    assert zipcode == l.zipcode

    l = Location(zipcode=zipcode)
    assert_allclose(lat_lng, (l.lat, l.lng), rtol=RTOL, atol=ATOL)
    assert station == l.station
    assert zipcode == l.zipcode

