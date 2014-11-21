from eemeter.consumption import Consumption
from eemeter.consumption import FuelType
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas
from eemeter.consumption import DateRangeException
from eemeter.consumption import InvalidFuelTypeException

from datetime import datetime

import pytest

EPSILON = 1e-6
##### Fixtures #####

@pytest.fixture(scope="module",
                params=[(0,"kWh",electricity,datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"kWh",electricity,datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"therms",electricity,datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"Btu",electricity,datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"Btu",natural_gas,datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"therms",natural_gas,datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"Btu",natural_gas,datetime(2000,1,1),datetime(2000,1,31),False),
                        (0,"kWh",electricity,datetime(2000,1,1),datetime(2000,1,31))])
def consumption_zero_one_month(request):
    return Consumption(*request.param)

@pytest.fixture(scope="session",params=[electricity,natural_gas,FuelType("NewType")])
def fuel_type(request):
    return request.param

##### Test cases #####

def test_consumption_has_correct_attributes(consumption_zero_one_month):
    assert consumption_zero_one_month.joules == 0
    assert isinstance(consumption_zero_one_month.fuel_type,FuelType)
    assert consumption_zero_one_month.start == datetime(2000,1,1)
    assert consumption_zero_one_month.end == datetime(2000,1,31)
    assert consumption_zero_one_month.estimated == False

def test_fuel_type(fuel_type):
    assert fuel_type.name == str(fuel_type)
    assert isinstance(fuel_type,FuelType)

def test_automatic_unit_conversion():
    btu_consumption = Consumption(1,"Btu",electricity,datetime(2000,1,1),datetime(2000,1,31),False)
    kwh_consumption = Consumption(1,"kWh",electricity,datetime(2000,1,1),datetime(2000,1,31),False)
    therm_consumption = Consumption(1,"therm",electricity,datetime(2000,1,1),datetime(2000,1,31),False)
    assert abs(1 - btu_consumption.to("Btu")) < EPSILON
    assert abs(1 - kwh_consumption.to("kWh")) < EPSILON
    assert abs(1 - therm_consumption.to("therm")) < EPSILON

def test_feasible_consumption_start_end():
    with pytest.raises(DateRangeException):
        Consumption(1,"Btu",electricity,datetime(2000,1,2),datetime(2000,1,1))

def test_consumption_invalid_fuel_type():
    with pytest.raises(InvalidFuelTypeException):
        Consumption(1,"Btu","Invalid",datetime(2000,1,1),datetime(2000,1,31))

def test_timedelta(consumption_zero_one_month):
    delta = consumption_zero_one_month.timedelta
    assert delta.days == 30

def test_consumption_has_representation(consumption_zero_one_month):
    assert "Consumption" in consumption_zero_one_month.__repr__()
