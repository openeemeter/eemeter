from eemeter.conversions import kwh_to_therm
from eemeter.conversions import therm_to_kwh
from eemeter.conversions import farenheight_to_celsius
from eemeter.conversions import celsius_to_farenheight
from eemeter.conversions import temp_to_hdd
from eemeter.conversions import temp_to_cdd

EPSILON = 1e-6

##### Tests #####

def test_kwh_to_therm():
    assert abs(kwh_to_therm(-1) - -29.3001111) < EPSILON
    assert abs(kwh_to_therm(0) - 0) < EPSILON
    assert abs(kwh_to_therm(1) - 29.3001111) < EPSILON

def test_therm_to_kwh():
    assert abs(therm_to_kwh(-1) - -0.0341295634) < EPSILON
    assert abs(therm_to_kwh(0) - 0) < EPSILON
    assert abs(therm_to_kwh(1) - 0.0341295634) < EPSILON

def test_farenheight_to_celsius():
    assert abs(farenheight_to_celsius(-1) - -18.33333333) < EPSILON
    assert abs(farenheight_to_celsius(0) - -17.77777778) < EPSILON
    assert abs(farenheight_to_celsius(1) - -17.22222222) < EPSILON

def test_celsius_to_farenheight():
    assert abs(celsius_to_farenheight(-1) - 30.2) < EPSILON
    assert abs(celsius_to_farenheight(0) - 32) < EPSILON
    assert abs(celsius_to_farenheight(1) - 33.8) < EPSILON

def test_temp_to_hdd():
    assert abs(temp_to_hdd(55,65) - 10) < EPSILON
    assert abs(temp_to_hdd(65,65) - 0) < EPSILON
    assert abs(temp_to_hdd(75,65) - 0) < EPSILON

def test_temp_to_cdd():
    assert abs(temp_to_cdd(55,65) - 0) < EPSILON
    assert abs(temp_to_cdd(65,65) - 0) < EPSILON
    assert abs(temp_to_cdd(75,65) - 10) < EPSILON
