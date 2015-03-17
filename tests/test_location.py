from eemeter.location import zipcode_to_lat_lng
from eemeter.location import lat_lng_to_zipcode
from eemeter.location import tmy3_to_lat_lng
from eemeter.location import lat_lng_to_tmy3
from eemeter.location import zipcode_to_tmy3
from eemeter.location import tmy3_to_zipcode
from eemeter.location import haversine

from numpy.testing import assert_allclose

RTOL = 1e-1
ATOL = 1e-1

def test_haversine():
    lat_lng_dists = [(0,0,0,0,0),
                     (76,1,76,1,0),
                     (76,1,76,361,0),
                     (0,0,0,90,10007.54339801),
                     (0,0,0,180,20015.08679602),
                     (0,-180,0,180,0),
                     (-90,0,90,0,20015.08679602),
                     (-90,0,90,180,20015.08679602),
                     ]

    for lat1,lng1,lat2,lng2,dist in lat_lng_dists:
        assert_allclose(haversine(lat1,lng1,lat2,lng2),dist,rtol=RTOL,atol=ATOL)

def test_zipcode_to_lat_lng():
    lat_lngs = [(41.8955360374983,-87.6217660821178),
               (34.1678563835543,-118.126220490392),
               (42.3769095103979,-71.1247640734676),
               (42.3594006437094,-87.8581578622419)]
    zipcodes = ["60611","91104","02138","60085"]
    for (lat,lng),zipcode in zip(lat_lngs,zipcodes):
        assert lat,lng == zipcode_to_lat_lng(zipcode)

def test_lat_lng_to_zipcode():
    lat_lngs = [(41.8955360374983,-87.6217660821178),
               (34.1678563835543,-118.126220490392),
               (42.3769095103979,-71.1247640734676),
               (42.3594006437094,-87.8581578622419)]
    zipcodes = ["60611","91104","02138","60085"]
    for (lat,lng),zipcode in zip(lat_lngs,zipcodes):
        assert zipcode == lat_lng_to_zipcode(lat,lng)

def test_tmy3_to_lat_lng():
    lat_lngs = [(41.8955360374983,-87.6217660821178),
               (34.1678563835543,-118.126220490392),
               (42.3769095103979,-71.1247640734676),
               (42.3594006437094,-87.8581578622419)]
    stations = ["725340","722880","725090","725347"]
    for (lat,lng),station in zip(lat_lngs,stations):
        assert lat,lng == tmy3_to_lat_lng(station)

def test_lat_lng_to_tmy3():
    lat_lngs = [(41.8955360374983,-87.6217660821178),
               (34.1678563835543,-118.126220490392),
               (42.3769095103979,-71.1247640734676),
               (42.3594006437094,-87.8581578622419)]
    stations = ["725340","722880","725090","725347"]
    for (lat,lng),station in zip(lat_lngs,stations):
        assert station == lat_lng_to_tmy3(lat,lng)

def test_zipcode_to_tmy3():
    zipcodes = ["60611","91104","02138","60085"]
    stations = ["725340","722880","725090","725347"]
    for zipcode,station in zip(zipcodes,stations):
        assert station == zipcode_to_tmy3(zipcode)

def test_tmy3_to_zipcode():
    zipcodes = ["97459","45433","55601","96740"]
    stations = ["726917","745700","727556","911975"]
    for zipcode,station in zip(zipcodes,stations):
        assert zipcode == tmy3_to_zipcode(station)

