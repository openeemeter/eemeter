from eemeter.units import EnergyUnit
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
    assert named_unit.abbreviation == str(named_unit)
    assert isinstance(named_unit.full_name,str)
    assert isinstance(named_unit.abbreviation,str)

def test_kilowatt_hour():
    assert "KilowattHour" == kWh.full_name
    assert "kWh" == kWh.abbreviation
    assert isinstance(kWh,EnergyUnit)

def test_british_thermal_unit():
    assert "BritishThermalUnit" == BTU.full_name
    assert "BTU" == BTU.abbreviation
    assert isinstance(BTU,EnergyUnit)

def test_therm():
    assert "Therm" == therm.full_name
    assert "therm" == therm.abbreviation
    assert isinstance(therm,EnergyUnit)
