from eemeter.usage import Usage
from eemeter.usage import FuelType
from eemeter.usage import electricity
from eemeter.usage import natural_gas
from eemeter.units import KilowattHour

import arrow

import pytest

##### Fixtures #####

@pytest.fixture(scope="module",
                params=[(0,KilowattHour,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,KilowattHour,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,KilowattHour,natural_gas,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0,KilowattHour,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime)])
def usage_zero_one_month(request):
    return Usage(*request.param)

@pytest.fixture(scope="session",params=[electricity,natural_gas,FuelType("NewType")])
def fuel_type(request):
    return request.param

##### Test cases #####

def test_usage_has_correct_attributes(usage_zero_one_month):
    assert usage_zero_one_month.usage == 0
    assert usage_zero_one_month.unit == KilowattHour
    assert isinstance(usage_zero_one_month.fuel_type,FuelType)
    assert usage_zero_one_month.start == arrow.get(2000,1,1).datetime
    assert usage_zero_one_month.end == arrow.get(2000,1,31).datetime
    assert usage_zero_one_month.estimated == False

def test_fuel_type(fuel_type):
    assert fuel_type.name == str(fuel_type)
    assert isinstance(fuel_type,FuelType)
