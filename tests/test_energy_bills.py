from eemeter.core import EnergyBill
from datetime import date
import pytest

##### Fixtures #####

# Helper fixture
@pytest.fixture(scope="module",
                params=[0,1,1000])
def _energy_bill(request):
    usage = request.param
    start_date = None
    end_date = None
    return EnergyBill(usage,start_date,end_date)

@pytest.fixture(scope="module",
                params=[(date(2014,1,1),date(2014,1,1),1),
                        (date(2014,1,1),date(2014,1,31),31)])
def valid_energy_bill(request,_energy_bill):
    _energy_bill.start_date, _energy_bill.end_date, days = request.param
    return _energy_bill,days

@pytest.fixture(scope="module",
                params=[(date(2014,1,1),date(2013,12,1))])
def energy_bill_with_invalid_date_range(request,_energy_bill):
    _energy_bill.start_date, _energy_bill.end_date = request.param
    return _energy_bill

@pytest.fixture(scope="module")
def energy_bill_with_invalid_dates(_energy_bill):
    return _energy_bill


##### Test cases #####

def test_valid_energy_bill_is_valid(valid_energy_bill):
    energy_bill,days = valid_energy_bill
    assert energy_bill.is_valid()

def test_energy_bill_with_invalid_date_range_is_invalid(energy_bill_with_invalid_date_range):
    assert not energy_bill_with_invalid_date_range.is_valid()

def test_energy_bill_with_invalid_dates_is_invalid(energy_bill_with_invalid_dates):
    assert not energy_bill_with_invalid_dates.is_valid()

def test_energy_bill_string_is_correct(valid_energy_bill):
    energy_bill,days = valid_energy_bill
    assert "Energy Bill " in str(energy_bill)
    assert "{}".format(energy_bill.end_date) in str(energy_bill)
    assert "{}".format(energy_bill.usage) in str(energy_bill)

def test_energy_bill_has_estimated_attribute(valid_energy_bill):
    energy_bill,days = valid_energy_bill
    assert hasattr(energy_bill,"estimated")
