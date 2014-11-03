from eemeter import units

import pytest

EPSILON = 1e-6

##### Fixtures #####

@pytest.fixture(scope="module",params=[
        units.KilowattHour,
        units.BritishThermalUnit,
        units.Therm])
def named_unit(request):
    return request.param

##### Tests #####

def test_named_unit(named_unit):
    assert named_unit.name == str(named_unit)
    assert isinstance(named_unit.name,str)
    assert isinstance(named_unit.abbr,str)
    assert isinstance(named_unit.description,str)

def test_kilowatthour():
    assert "KilowattHour" == units.KilowattHour.name
    assert "kWh" == units.KilowattHour.abbr
    assert "Unit of energy" == units.KilowattHour.description

def test_british_thermal_unit():
    assert "BritishThermalUnit" == units.BritishThermalUnit.name
    assert "BTU" == units.BritishThermalUnit.abbr
    assert "Unit of energy" == units.BritishThermalUnit.description

def test_therm():
    assert "Therm" == units.Therm.name
    assert "therm" == units.Therm.abbr
    assert "Unit of energy" == units.Therm.description

def test_kwh_to_therm():
    assert abs(units.kwh_to_therm(-1) - -29.3001111) < EPSILON
    assert abs(units.kwh_to_therm(0) - 0) < EPSILON
    assert abs(units.kwh_to_therm(1) - 29.3001111) < EPSILON

def test_therm_to_kwh():
    assert abs(units.therm_to_kwh(-1) - -0.0341295634) < EPSILON
    assert abs(units.therm_to_kwh(0) - 0) < EPSILON
    assert abs(units.therm_to_kwh(1) - 0.0341295634) < EPSILON

def test_farenheight_to_celsius():
    assert abs(units.farenheight_to_celsius(-1) - -18.33333333) < EPSILON
    assert abs(units.farenheight_to_celsius(0) - -17.77777778) < EPSILON
    assert abs(units.farenheight_to_celsius(1) - -17.22222222) < EPSILON

def test_celsius_to_farenheight():
    assert abs(units.celsius_to_farenheight(-1) - 30.2) < EPSILON
    assert abs(units.celsius_to_farenheight(0) - 32) < EPSILON
    assert abs(units.celsius_to_farenheight(1) - 33.8) < EPSILON

def test_temp_to_hdd():
    assert abs(units.temp_to_hdd(55,65) - 10) < EPSILON
    assert abs(units.temp_to_hdd(65,65) - 0) < EPSILON
    assert abs(units.temp_to_hdd(75,65) - 0) < EPSILON

def test_temp_to_cdd():
    assert abs(units.temp_to_cdd(55,65) - 0) < EPSILON
    assert abs(units.temp_to_cdd(65,65) - 0) < EPSILON
    assert abs(units.temp_to_cdd(75,65) - 10) < EPSILON
