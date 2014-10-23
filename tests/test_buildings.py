from eemeter.core import EnergyBill, Building
from datetime import date, timedelta

import pytest

##### Fixtures #####

@pytest.fixture(scope="module")
def building_that_meets_criteria(request):
    energy_bills = [EnergyBill(0,date(2012,1,1),date(2012,1,31)),
                    EnergyBill(0,date(2012,2,1),date(2012,2,28)),
                    EnergyBill(0,date(2012,3,1),date(2012,3,31)),
                    EnergyBill(0,date(2012,4,1),date(2012,4,30)),
                    EnergyBill(0,date(2012,5,1),date(2012,5,31)),
                    EnergyBill(0,date(2012,6,1),date(2012,6,30)),
                    EnergyBill(0,date(2012,7,1),date(2012,7,31)),
                    EnergyBill(0,date(2012,8,1),date(2012,8,31)),
                    EnergyBill(0,date(2012,9,1),date(2012,9,30)),
                    EnergyBill(0,date(2012,10,1),date(2012,10,31)),
                    EnergyBill(0,date(2012,11,1),date(2012,11,30)),
                    EnergyBill(0,date(2012,12,1),date(2012,12,31)),
                    EnergyBill(0,date(2013,1,1),date(2013,1,31)),
                    EnergyBill(0,date(2013,2,1),date(2013,2,28)),
                    EnergyBill(0,date(2013,3,1),date(2013,3,31)),
                    EnergyBill(0,date(2013,4,1),date(2013,4,30)),
                    EnergyBill(0,date(2013,5,1),date(2013,5,31)),
                    EnergyBill(0,date(2013,6,1),date(2013,6,30)),
                    EnergyBill(0,date(2013,7,1),date(2013,7,31)),
                    EnergyBill(0,date(2013,8,1),date(2013,8,31)),
                    EnergyBill(0,date(2013,9,1),date(2013,9,30)),
                    EnergyBill(0,date(2013,10,1),date(2013,10,31)),
                    EnergyBill(0,date(2013,11,1),date(2013,11,30)),
                    EnergyBill(0,date(2013,12,1),date(2013,12,31))]
    return Building(energy_bills)

@pytest.fixture(scope="module")
def building_with_old_bills(request):
    energy_bills = [EnergyBill(0,date(2012,8,1),date(2012,8,31)),
                    EnergyBill(0,date(2012,9,1),date(2012,9,30)),
                    EnergyBill(0,date(2012,10,1),date(2012,10,31)),
                    EnergyBill(0,date(2012,11,1),date(2012,11,30)),
                    EnergyBill(0,date(2012,12,1),date(2012,12,31)),
                    EnergyBill(0,date(2012,1,1),date(2012,1,31)),
                    EnergyBill(0,date(2012,2,1),date(2012,2,28)),
                    EnergyBill(0,date(2012,3,1),date(2012,3,31)),
                    EnergyBill(0,date(2012,4,1),date(2012,4,30)),
                    EnergyBill(0,date(2012,5,1),date(2012,5,31)),
                    EnergyBill(0,date(2012,6,1),date(2012,6,30)),
                    EnergyBill(0,date(2012,7,1),date(2012,7,31)),
                    EnergyBill(0,date(2012,8,1),date(2012,8,31)),
                    EnergyBill(0,date(2012,9,1),date(2012,9,30)),
                    EnergyBill(0,date(2012,10,1),date(2012,10,31)),
                    EnergyBill(0,date(2012,11,1),date(2012,11,30)),
                    EnergyBill(0,date(2012,12,1),date(2012,12,31))]
    return Building(energy_bills)

@pytest.fixture(scope="module")
def building_with_too_few_bills(request):
    return Building([])

##### Tests #####

def test_building_has_bills(building_that_meets_criteria):
    assert len(building_that_meets_criteria.energy_bills) >= 0

def test_building_string(building_that_meets_criteria):
    assert "Building (24 bills)" == str(building_that_meets_criteria)

def test_building_meets_criteria(building_that_meets_criteria):
    assert building_that_meets_criteria.meets_criteria()
