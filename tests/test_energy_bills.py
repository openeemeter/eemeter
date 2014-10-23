from eemeter.core import EnergyBill
from datetime import date
import pytest

##### Fixtures #####

# Helper fixture
@pytest.fixture(scope="module",
                params=[0,1,1000,-1000])
def _energy_bill(request):
    usage = request.param
    start_date = None
    end_date = None
    return EnergyBill(usage,start_date,end_date)

@pytest.fixture(scope="module",
                params=[(date(2014,1,1),date(2014,1,1),1),
                        (date(2014,1,1),date(2014,1,31),31)])
def energy_bill_with_valid_date(request,_energy_bill):
    _energy_bill.start_date, _energy_bill.end_date, days = request.param
    return _energy_bill,days

@pytest.fixture(scope="module",
                params=[(date(2014,1,1),date(2013,12,1))])
def energy_bill_with_invalid_date(request,_energy_bill):
    _energy_bill.start_date, _energy_bill.end_date = request.param
    return _energy_bill

##### Test cases #####

def test_energy_bill_has_valid_date(energy_bill_with_valid_date):
    energy_bill,days = energy_bill_with_valid_date
    assert energy_bill.start_date <= \
            energy_bill.end_date
    assert energy_bill.days() == days

def test_energy_bill_has_invalid_date(energy_bill_with_invalid_date):
    with pytest.raises(AssertionError):
        energy_bill_with_invalid_date.days()
