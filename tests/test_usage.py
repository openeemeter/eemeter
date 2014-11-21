from eemeter.usage import Usage
from eemeter.usage import FuelType
from eemeter.usage import electricity
from eemeter.usage import natural_gas
from eemeter.usage import DateRangeException
from eemeter.usage import InvalidFuelTypeException

from eemeter.units import EnergyUnit
from eemeter.units import kWh
from eemeter.units import therm
from eemeter.units import BTU

import arrow

import pytest

EPSILON = 1e-6
##### Fixtures #####

@pytest.fixture(scope="module",
                params=[(0,kWh,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,kWh,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,therm,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,BTU,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,kWh,natural_gas,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,therm,natural_gas,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0.0,BTU,natural_gas,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False),
                        (0,kWh,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime)])
def usage_zero_one_month(request):
    return Usage(*request.param)

@pytest.fixture(scope="session",params=[electricity,natural_gas,FuelType("NewType")])
def fuel_type(request):
    return request.param

##### Test cases #####

def test_usage_has_correct_attributes(usage_zero_one_month):
    assert usage_zero_one_month.BTU == 0
    assert isinstance(usage_zero_one_month.fuel_type,FuelType)
    assert usage_zero_one_month.start == arrow.get(2000,1,1).datetime
    assert usage_zero_one_month.end == arrow.get(2000,1,31).datetime
    assert usage_zero_one_month.estimated == False

def test_fuel_type(fuel_type):
    assert fuel_type.name == str(fuel_type)
    assert isinstance(fuel_type,FuelType)

def test_automatic_unit_conversion():
    btu_usage = Usage(1,BTU,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False)
    kwh_usage = Usage(1,kWh,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False)
    therm_usage = Usage(1,therm,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False)
    assert abs(1 - btu_usage.to(BTU)) < EPSILON
    assert abs(1 - kwh_usage.to(kWh)) < EPSILON
    assert abs(1 - therm_usage.to(therm)) < EPSILON
    assert abs(2.9307107e-4 - btu_usage.to(kWh)) < EPSILON
    assert abs(0.0341295634 - kwh_usage.to(therm)) < EPSILON
    assert abs(9.9976129e4 - therm_usage.to(BTU)) < EPSILON

def test_feasible_usage_start_end():
    with pytest.raises(DateRangeException):
        Usage(1,BTU,electricity,arrow.get(2000,1,2).datetime,arrow.get(2000,1,1).datetime)

def test_usage_invalid_fuel_type():
    with pytest.raises(InvalidFuelTypeException):
        Usage(1,BTU,"Invalid",arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime)

def test_timedelta(usage_zero_one_month):
    delta = usage_zero_one_month.timedelta
    assert delta.days == 30

def test_usage_has_representation(usage_zero_one_month):
    assert "Usage" in usage_zero_one_month.__repr__()
