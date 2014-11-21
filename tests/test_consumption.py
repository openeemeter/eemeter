from eemeter.consumption import Consumption
from eemeter.consumption import FuelType
from eemeter.consumption import electricity
from eemeter.consumption import natural_gas
from eemeter.consumption import DateRangeException
from eemeter.consumption import InvalidFuelTypeException

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
def consumption_zero_one_month(request):
    return Consumption(*request.param)

@pytest.fixture(scope="session",params=[electricity,natural_gas,FuelType("NewType")])
def fuel_type(request):
    return request.param

##### Test cases #####

def test_consumption_has_correct_attributes(consumption_zero_one_month):
    assert consumption_zero_one_month.BTU == 0
    assert isinstance(consumption_zero_one_month.fuel_type,FuelType)
    assert consumption_zero_one_month.start == arrow.get(2000,1,1).datetime
    assert consumption_zero_one_month.end == arrow.get(2000,1,31).datetime
    assert consumption_zero_one_month.estimated == False

def test_fuel_type(fuel_type):
    assert fuel_type.name == str(fuel_type)
    assert isinstance(fuel_type,FuelType)

def test_automatic_unit_conversion():
    btu_consumption = Consumption(1,BTU,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False)
    kwh_consumption = Consumption(1,kWh,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False)
    therm_consumption = Consumption(1,therm,electricity,arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime,False)
    assert abs(1 - btu_consumption.to(BTU)) < EPSILON
    assert abs(1 - kwh_consumption.to(kWh)) < EPSILON
    assert abs(1 - therm_consumption.to(therm)) < EPSILON
    assert abs(2.9307107e-4 - btu_consumption.to(kWh)) < EPSILON
    assert abs(0.0341295634 - kwh_consumption.to(therm)) < EPSILON
    assert abs(9.9976129e4 - therm_consumption.to(BTU)) < EPSILON

def test_feasible_consumption_start_end():
    with pytest.raises(DateRangeException):
        Consumption(1,BTU,electricity,arrow.get(2000,1,2).datetime,arrow.get(2000,1,1).datetime)

def test_consumption_invalid_fuel_type():
    with pytest.raises(InvalidFuelTypeException):
        Consumption(1,BTU,"Invalid",arrow.get(2000,1,1).datetime,arrow.get(2000,1,31).datetime)

def test_timedelta(consumption_zero_one_month):
    delta = consumption_zero_one_month.timedelta
    assert delta.days == 30

def test_consumption_has_representation(consumption_zero_one_month):
    assert "Consumption" in consumption_zero_one_month.__repr__()
