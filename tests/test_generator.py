from eemeter.generator import ConsumptionGenerator
from eemeter.consumption import DatetimePeriod
from fixtures.weather import gsod_722880_2012_2014_weather_source
from eemeter.consumption import electricity

import pytest
from datetime import datetime

@pytest.fixture
def periods_one_year():
    return [DatetimePeriod(datetime(2012,1,1),datetime(2012,2,1)),
            DatetimePeriod(datetime(2012,2,1),datetime(2012,3,1)),
            DatetimePeriod(datetime(2012,3,1),datetime(2012,4,1)),
            DatetimePeriod(datetime(2012,4,1),datetime(2012,5,1)),
            DatetimePeriod(datetime(2012,5,1),datetime(2012,6,1)),
            DatetimePeriod(datetime(2012,6,1),datetime(2012,7,1)),
            DatetimePeriod(datetime(2012,7,1),datetime(2012,8,1)),
            DatetimePeriod(datetime(2012,8,1),datetime(2012,9,1)),
            DatetimePeriod(datetime(2012,9,1),datetime(2012,10,1)),
            DatetimePeriod(datetime(2012,10,1),datetime(2012,11,1)),
            DatetimePeriod(datetime(2012,11,1),datetime(2012,12,1)),
            DatetimePeriod(datetime(2012,12,1),datetime(2013,1,1))]

#@pytest.mark.slow
def test_generator():
    gen = ConsumptionGenerator(electricity, "J", "degF", 65, 1, 75, 1, gsod_722880_2012_2014_weather_source())
    consumptions = gen.generate(periods_one_year())