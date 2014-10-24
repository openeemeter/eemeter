from eemeter import units

epsilon = 1e-6

def test_kwh_to_therm():
    assert abs(units.kwh_to_therm(-1) - -29.3001111) < epsilon
    assert abs(units.kwh_to_therm(0) - 0) < epsilon
    assert abs(units.kwh_to_therm(1) - 29.3001111) < epsilon

def test_therm_to_kwh():
    assert abs(units.therm_to_kwh(-1) - -0.0341295634) < epsilon
    assert abs(units.therm_to_kwh(0) - 0) < epsilon
    assert abs(units.therm_to_kwh(1) - 0.0341295634) < epsilon

def test_farenheight_to_celcius():
    assert abs(units.farenheight_to_celcius(-1) - -18.33333333) < epsilon
    assert abs(units.farenheight_to_celcius(0) - -17.77777778) < epsilon
    assert abs(units.farenheight_to_celcius(1) - -17.22222222) < epsilon

def test_celcius_to_farenheight():
    assert abs(units.celcius_to_farenheight(-1) - 30.2) < epsilon
    assert abs(units.celcius_to_farenheight(0) - 32) < epsilon
    assert abs(units.celcius_to_farenheight(1) - 33.8) < epsilon

def test_temp_to_hdd():
    assert abs(units.temp_to_hdd(55,65) - 10) < epsilon
    assert abs(units.temp_to_hdd(65,65) - 0) < epsilon
    assert abs(units.temp_to_hdd(75,65) - 0) < epsilon

def test_temp_to_cdd():
    assert abs(units.temp_to_cdd(55,65) - 0) < epsilon
    assert abs(units.temp_to_cdd(65,65) - 0) < epsilon
    assert abs(units.temp_to_cdd(75,65) - 10) < epsilon
