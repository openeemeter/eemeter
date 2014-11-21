from eemeter.consumption import Consumption
from eemeter.consumption import electricity

from datetime import datetime

import pytest

EPSILON = 10e-6

##### Fixtures #####

@pytest.fixture
def consumption_one_year_electricity():
    return [Consumption(1000,"kWh",electricity,datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(1100,"kWh",electricity,datetime(2012,2,1),datetime(2012,3,1)),
            Consumption(1200,"kWh",electricity,datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(1300,"kWh",electricity,datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(1400,"kWh",electricity,datetime(2012,5,1),datetime(2012,6,1)),
            Consumption(1500,"kWh",electricity,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(1400,"kWh",electricity,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(1300,"kWh",electricity,datetime(2012,8,1),datetime(2012,9,1)),
            Consumption(1200,"kWh",electricity,datetime(2012,9,1),datetime(2012,10,1)),
            Consumption(1100,"kWh",electricity,datetime(2012,10,1),datetime(2012,11,1)),
            Consumption(1000,"kWh",electricity,datetime(2012,11,1),datetime(2012,12,1)),
            Consumption(900,"kWh",electricity,datetime(2012,12,1),datetime(2013,1,1))]

@pytest.fixture
def consumption_one_summer_electricity():
    return [Consumption(1600,"kWh",electricity,datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(1700,"kWh",electricity,datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(1800,"kWh",electricity,datetime(2012,8,1),datetime(2012,9,1))]

@pytest.fixture
def degrees_f_one_year():
    return [20,15,20,35,55,65,80,80,60,45,40,30]

##### Tests #####

