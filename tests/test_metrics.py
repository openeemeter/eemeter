from eemeter.consumption import Consumption
from eemeter.consumption import electricity
from eemeter.units import kWh
from eemeter.units import BTU

import arrow

import pytest

EPSILON = 10e-6

##### Fixtures #####

@pytest.fixture
def consumption_one_year_electricity():
    return [Consumption(1000,kWh,electricity,arrow.get(2012,1,1).datetime,arrow.get(2012,2,1).datetime),
            Consumption(1100,kWh,electricity,arrow.get(2012,2,1).datetime,arrow.get(2012,3,1).datetime),
            Consumption(1200,kWh,electricity,arrow.get(2012,3,1).datetime,arrow.get(2012,4,1).datetime),
            Consumption(1300,kWh,electricity,arrow.get(2012,4,1).datetime,arrow.get(2012,5,1).datetime),
            Consumption(1400,kWh,electricity,arrow.get(2012,5,1).datetime,arrow.get(2012,6,1).datetime),
            Consumption(1500,kWh,electricity,arrow.get(2012,6,1).datetime,arrow.get(2012,7,1).datetime),
            Consumption(1400,kWh,electricity,arrow.get(2012,7,1).datetime,arrow.get(2012,8,1).datetime),
            Consumption(1300,kWh,electricity,arrow.get(2012,8,1).datetime,arrow.get(2012,9,1).datetime),
            Consumption(1200,kWh,electricity,arrow.get(2012,9,1).datetime,arrow.get(2012,10,1).datetime),
            Consumption(1100,kWh,electricity,arrow.get(2012,10,1).datetime,arrow.get(2012,11,1).datetime),
            Consumption(1000,kWh,electricity,arrow.get(2012,11,1).datetime,arrow.get(2012,12,1).datetime),
            Consumption(900,kWh,electricity,arrow.get(2012,12,1).datetime,arrow.get(2013,1,1).datetime)]

@pytest.fixture
def consumption_one_summer_electricity():
    return [Consumption(1600,kWh,electricity,arrow.get(2012,6,1).datetime,arrow.get(2012,7,1).datetime),
            Consumption(1700,kWh,electricity,arrow.get(2012,7,1).datetime,arrow.get(2012,8,1).datetime),
            Consumption(1800,kWh,electricity,arrow.get(2012,8,1).datetime,arrow.get(2012,9,1).datetime)]

@pytest.fixture
def degrees_f_one_year():
    return [20,15,20,35,55,65,80,80,60,45,40,30]

##### Tests #####

