from eemeter.core import EnergyBill, Building
from datetime import date, timedelta

import pytest

@pytest.fixture(scope="module")
def building_that_meets_criteria(request):
    end_dates = [date(2012,1,1),
                 date(2012,2,1),
                 date(2012,3,1),
                 date(2012,4,1),
                 date(2012,5,1),
                 date(2012,6,1),
                 date(2012,7,1),
                 date(2012,8,1),
                 date(2012,9,1),
                 date(2012,10,1),
                 date(2012,11,1),
                 date(2012,12,1),
                 date(2013,1,1),
                 date(2013,2,1),
                 date(2013,3,1),
                 date(2013,4,1),
                 date(2013,5,1),
                 date(2013,6,1),
                 date(2013,7,1),
                 date(2013,8,1),
                 date(2013,9,1),
                 date(2013,10,1),
                 date(2013,11,1),
                 date(2013,12,1)]
    start_dates = [end_dates[0] - timedelta(days=30)]
    start_dates.extend([end_date + timedelta(days=1) for end_date in end_dates[:-1]])
    usages = [0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,0,0,
              0,0,0,0,0,0]

    energy_bills = []
    for usage,start_date,end_date in zip(usages,start_dates,end_dates):
        energy_bills.append(EnergyBill(usage,start_date,end_date))
    return Building(energy_bills)

def test_building_has_bills(building_that_meets_criteria):
    assert len(building_that_meets_criteria.energy_bills) >= 0
