from eemeter.location import zipcode_to_lat_lng
from eemeter.location import lat_lng_to_zipcode
from eemeter.location import station_to_lat_lng
from eemeter.location import lat_lng_to_station
from eemeter.location import zipcode_to_station
from eemeter.location import station_to_zipcode
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
    ((41.8955360374983,-87.6217660821178),"60611","725340","60638",(41.783,-87.75)),
    ((34.1678563835543,-118.126220490392),"91104","722880","91504",(34.200,-118.35)),
    ((42.3769095103979,-71.1247640734676),"02138","725090","02128",(42.367,-71.017)),
    ((42.3594006437094,-87.8581578622419),"60085","725347","60087",(42.417,-87.867))
    ])
def lat_lng_zipcode_station(request):
    return request.param

##### TESTS #####

def test_haversine(lat_lng_dist):
    lat1,lng1,lat2,lng2,dist = lat_lng_dist
    assert_allclose(haversine(lat1,lng1,lat2,lng2),dist,rtol=RTOL,atol=ATOL)

def test_zipcode_to_lat_lng(lat_lng_zipcode_station):
    (lat,lng),zipcode,station,station_zip,(st_lat,st_lng) = lat_lng_zipcode_station
    assert lat,lng == zipcode_to_lat_lng(zipcode)

def test_lat_lng_to_zipcode(lat_lng_zipcode_station):
    (lat,lng),zipcode,station,station_zip,(st_lat,st_lng) = lat_lng_zipcode_station
    assert zipcode == lat_lng_to_zipcode(lat,lng)

def test_station_to_lat_lng(lat_lng_zipcode_station):
    (lat,lng),zipcode,station,station_zip,(st_lat,st_lng) = lat_lng_zipcode_station
    assert st_lat,st_lng == station_to_lat_lng(station)

def test_lat_lng_to_station(lat_lng_zipcode_station):
    (lat,lng),zipcode,station,station_zip,(st_lat,st_lng) = lat_lng_zipcode_station
    assert station == lat_lng_to_station(lat,lng)

def test_zipcode_to_station(lat_lng_zipcode_station):
    (lat,lng),zipcode,station,station_zip,(st_lat,st_lng) = lat_lng_zipcode_station
    assert station == zipcode_to_station(zipcode)

def test_station_to_zipcode(lat_lng_zipcode_station):
    (lat,lng),zipcode,station,station_zip,(st_lat,st_lng) = lat_lng_zipcode_station
    assert station_zip == station_to_zipcode(station)

def test_location_init(lat_lng_zipcode_station):
    (lat,lng),zipcode,station,station_zip,(st_lat,st_lng) = lat_lng_zipcode_station

    l = Location(lat_lng=(lat,lng))
    assert lat == l.lat
    assert lng == l.lng
    assert station == l.station
    assert zipcode == l.zipcode

    l = Location(zipcode=zipcode)
    assert lat == l.lat
    assert lng == l.lng
    assert station == l.station
    assert zipcode == l.zipcode

    l = Location(station=station)
    assert st_lat == l.lat
    assert st_lng == l.lng
    assert station == l.station
    assert station_zip == l.zipcode
