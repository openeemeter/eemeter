from numpy.testing import assert_allclose

from eemeter.weather.location import (
    haversine,
    lat_lng_to_usaf_station,
    lat_lng_to_tmy3_station,
    lat_lng_to_zipcode,
    lat_lng_to_climate_zone,
    usaf_station_to_lat_lng,
    usaf_station_to_zipcodes,
    usaf_station_to_climate_zone,
    tmy3_station_to_lat_lng,
    tmy3_station_to_zipcodes,
    tmy3_station_to_climate_zone,
    zipcode_to_lat_lng,
    zipcode_to_usaf_station,
    zipcode_to_tmy3_station,
    zipcode_to_climate_zone,
    climate_zone_to_zipcodes,
    climate_zone_to_usaf_stations,
    climate_zone_to_tmy3_stations,
    zipcode_is_supported,
    usaf_station_is_supported,
    tmy3_station_is_supported,
    climate_zone_is_supported,
)


def test_haversine():
    assert_allclose(haversine(0, 0, 20, 20), 3112.445040)


def test_lat_lng_to_usaf_station():
    assert lat_lng_to_usaf_station(40, -100) == '725625'


def test_lat_lng_to_tmy3_station():
    assert lat_lng_to_tmy3_station(45, -90) == '726463'


def test_lat_lng_to_zipcode():
    assert lat_lng_to_zipcode(42, -120) == '96112'


def test_lat_lng_to_climate_zone():
    assert lat_lng_to_climate_zone(43, -95) == '6|A|Cold'


def test_usaf_station_to_lat_lng():
    assert_allclose(usaf_station_to_lat_lng('720655'), [28.867, -82.571])


def test_usaf_station_to_zipcodes():
    assert '83204' in usaf_station_to_zipcodes('725780')


def test_usaf_station_to_climate_zone():
    assert usaf_station_to_climate_zone('724677') == '7|NA|Very Cold'


def test_tmy3_station_to_lat_lng():
    assert_allclose(tmy3_station_to_lat_lng('745700'), [39.833, -84.05])


def test_tmy3_station_to_zipcodes():
    assert '81058' in tmy3_station_to_zipcodes('724635')


def test_tmy3_station_to_climate_zone():
    assert tmy3_station_to_climate_zone('725065') == '5|A|Cold'


def test_zipcode_to_lat_lng():
    assert_allclose(zipcode_to_lat_lng('16701'), [41.917904, -78.762944])


def test_zipcode_to_usaf_station():
    assert zipcode_to_usaf_station('82440') == '726700'
    assert zipcode_to_usaf_station('94403') == '994041'


def test_zipcode_to_tmy3_station():
    assert zipcode_to_tmy3_station('19975') == '745966'
    assert zipcode_to_tmy3_station('94403') == '724940'


def test_zipcode_to_climate_zone():
    assert zipcode_to_climate_zone('81050') == '4|B|Mixed-Dry'


def test_climate_zone_to_zipcodes():
    assert '95968' in climate_zone_to_zipcodes('CA_11')


def test_climate_zone_to_usaf_stations():
    assert '724955' in climate_zone_to_usaf_stations('CA_02')


def test_climate_zone_to_tmy3_stations():
    assert '725555' in climate_zone_to_tmy3_stations('5|A|Cold')


def test_zipcode_is_supported():
    assert zipcode_is_supported('81050') is True


def test_usaf_station_is_supported():
    assert usaf_station_is_supported('726700') is True


def test_tmy3_station_is_supported():
    assert tmy3_station_is_supported('724397') is True


def test_climate_zone_is_supported():
    assert climate_zone_is_supported('4|B|Mixed-Dry') is True
