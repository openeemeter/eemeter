from eemeter.units import BTU
from eemeter.units import kWh
from eemeter.units import therm

import pytest

EPSILON = 1e-6

##### Fixtures #####

@pytest.fixture(scope="module",params=[BTU,kWh,therm])
def named_unit(request):
    return request.param

##### Tests #####

def test_named_unit(named_unit):
    assert named_unit.name == str(named_unit)
    assert isinstance(named_unit.name,str)
    assert isinstance(named_unit.abbr,str)
    assert isinstance(named_unit.description,str)

def test_kilowatt_hour():
    assert "KilowattHour" == kWh.name
    assert "kWh" == kWh.abbr
    assert "Unit of energy" == kWh.description

def test_british_thermal_unit():
    assert "BritishThermalUnit" == BTU.name
    assert "BTU" == BTU.abbr
    assert "Unit of energy" == BTU.description

def test_therm():
    assert "Therm" == therm.name
    assert "therm" == therm.abbr
    assert "Unit of energy" == therm.description
