from eemeter.energy import Usage
from eemeter.units import KilowattHour

import arrow

import pytest


##### Fixtures #####

@pytest.fixture(scope="module",
                params=[(0,KilowattHour,arrow.get(2000,1,1),arrow.get(2000,1,31)),
                        (0.0,KilowattHour,arrow.get(2000,1,1),arrow.get(2000,1,31))])
def usage_zero_month(request):
    return Usage(*request.param)

##### Test cases #####

def test_usage_has_correct_attributes(usage_zero_month):
    assert usage_zero_month.usage == 0
    assert usage_zero_month.unit == KilowattHour
    assert usage_zero_month.start == arrow.get(2000,1,1)
    assert usage_zero_month.end == arrow.get(2000,1,31)
