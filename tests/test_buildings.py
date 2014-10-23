from eemeter.core import EnergyBill, Building
from datetime import date, timedelta

import pytest

##### Fixtures #####

@pytest.fixture(scope="module")
def building_that_meets_criteria(request):
    energy_bills = [EnergyBill(1,date(2012,1,1),date(2012,1,31)),
                    EnergyBill(1,date(2012,2,1),date(2012,2,28)),
                    EnergyBill(1,date(2012,3,1),date(2012,3,31)),
                    EnergyBill(1,date(2012,4,1),date(2012,4,30)),
                    EnergyBill(1,date(2012,5,1),date(2012,5,31)),
                    EnergyBill(1,date(2012,6,1),date(2012,6,30)),
                    EnergyBill(1,date(2012,7,1),date(2012,7,31)),
                    EnergyBill(1,date(2012,8,1),date(2012,8,31)),
                    EnergyBill(1,date(2012,9,1),date(2012,9,30)),
                    EnergyBill(1,date(2012,10,1),date(2012,10,31)),
                    EnergyBill(1,date(2012,11,1),date(2012,11,30)),
                    EnergyBill(1,date(2012,12,1),date(2012,12,31)),
                    EnergyBill(1,date(2013,1,1),date(2013,1,31)),
                    EnergyBill(1,date(2013,2,1),date(2013,2,28)),
                    EnergyBill(1,date(2013,3,1),date(2013,3,31)),
                    EnergyBill(1,date(2013,4,1),date(2013,4,30)),
                    EnergyBill(1,date(2013,5,1),date(2013,5,31)),
                    EnergyBill(1,date(2013,6,1),date(2013,6,30)),
                    EnergyBill(1,date(2013,7,1),date(2013,7,31)),
                    EnergyBill(1,date(2013,8,1),date(2013,8,31)),
                    EnergyBill(1,date(2013,9,1),date(2013,9,30)),
                    EnergyBill(1,date(2013,10,1),date(2013,10,31)),
                    EnergyBill(1,date(2013,11,1),date(2013,11,30)),
                    EnergyBill(1,date(2013,12,1),date(2013,12,31))]
    return Building(energy_bills)

@pytest.fixture(scope="module")
def building_with_old_bills(request):
    energy_bills = [EnergyBill(0,date(2011,8,1),date(2011,8,31)),
                    EnergyBill(0,date(2011,9,1),date(2011,9,30)),
                    EnergyBill(0,date(2011,10,1),date(2011,10,31)),
                    EnergyBill(0,date(2011,11,1),date(2011,11,30)),
                    EnergyBill(0,date(2011,12,1),date(2011,12,31)),
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

@pytest.fixture(scope="module")
def building_with_future_bill(request):
    return Building([EnergyBill(0,date(3000,1,1),date(3000,1,1))])

##### Tests #####

def test_building_has_bills(building_that_meets_criteria):
    assert len(building_that_meets_criteria.energy_bills) >= 0

def test_building_string(building_that_meets_criteria):
    assert "Building (24 bills)" == str(building_that_meets_criteria)

def test_building_meets_criteria(building_that_meets_criteria):
    assert building_that_meets_criteria.meets_calibration_criteria()

def test_building_with_too_few_bills_doesnt_meet_criteria(building_with_too_few_bills):
    assert not building_with_too_few_bills.meets_calibration_criteria()

def test_building_with_future_bill_doesnt_meet_criteria(building_with_future_bill):
    assert not building_with_future_bill.meets_calibration_criteria()

def test_building_most_recent_bill(building_that_meets_criteria,
                                   building_with_old_bills,
                                   building_with_too_few_bills):
    assert building_that_meets_criteria.most_recent_energy_bill().end_date == date(2013,12,31)
    assert building_with_old_bills.most_recent_energy_bill().end_date == date(2012,12,31)
    assert building_with_too_few_bills.most_recent_energy_bill() == None

def test_building_most_recent_bill(building_that_meets_criteria):
    assert building_that_meets_criteria.total_usage() == 24
