import pytest

from eemeter.consumption import Consumption,ConsumptionHistory

from datetime import datetime

@pytest.fixture
def consumption_history_1():
    c_list = [Consumption(687600000,"J","electricity",datetime(2012,9,26),datetime(2012,10,24)),
            Consumption(874800000,"J","electricity",datetime(2012,10,24),datetime(2012,11,21)),
            Consumption(1332000000,"J","electricity",datetime(2012,11,21),datetime(2012,12,27)),
            Consumption(1454400000,"J","electricity",datetime(2012,12,27),datetime(2013,1,29)),
            Consumption(1155600000,"J","electricity",datetime(2013,1,29),datetime(2013,2,26)),
            Consumption(1195200000,"J","electricity",datetime(2013,2,26),datetime(2013,3,27)),
            Consumption(1033200000,"J","electricity",datetime(2013,3,27),datetime(2013,4,25)),
            Consumption(752400000,"J","electricity",datetime(2013,4,25),datetime(2013,5,23)),
            Consumption(889200000,"J","electricity",datetime(2013,5,23),datetime(2013,6,22)),
            Consumption(3434400000,"J","electricity",datetime(2013,6,22),datetime(2013,7,26)),
            Consumption(828000000,"J","electricity",datetime(2013,7,26),datetime(2013,8,22)),
            Consumption(2217600000,"J","electricity",datetime(2013,8,22),datetime(2013,9,25)),
            Consumption(680400000,"J","electricity",datetime(2013,9,25),datetime(2013,10,23)),
            Consumption(1062000000,"J","electricity",datetime(2013,10,23),datetime(2013,11,22)),
            Consumption(1720800000,"J","electricity",datetime(2013,11,22),datetime(2013,12,27)),
            Consumption(1915200000,"J","electricity",datetime(2013,12,27),datetime(2014,1,30)),
            Consumption(1458000000,"J","electricity",datetime(2014,1,30),datetime(2014,2,27)),
            Consumption(1332000000,"J","electricity",datetime(2014,2,27),datetime(2014,3,29)),
            Consumption(954000000,"J","electricity",datetime(2014,3,29),datetime(2014,4,26)),
            Consumption(842400000,"J","electricity",datetime(2014,4,26),datetime(2014,5,28)),
            Consumption(1220400000,"J","electricity",datetime(2014,5,28),datetime(2014,6,25)),
            Consumption(1702800000,"J","electricity",datetime(2014,6,25),datetime(2014,7,25)),
            Consumption(1375200000,"J","electricity",datetime(2014,7,25),datetime(2014,8,23)),
            Consumption(1623600000,"J","electricity",datetime(2014,8,23),datetime(2014,9,25))]
    return ConsumptionHistory(c_list)
