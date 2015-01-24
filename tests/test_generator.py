from eemeter.generator import ConsumptionGenerator
from eemeter.consumption import DatetimePeriod
from eemeter.consumption import electricity

from fixtures.weather import gsod_722880_2012_2014_weather_source
from helpers import arrays_similar

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

@pytest.mark.slow
def test_generator(periods_one_year):
    gen = ConsumptionGenerator(electricity, "J", "degF", 65, 1, 75, 1)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods_one_year)
    consumption_joules = [c.to("J") for c in consumptions]
    assert len(consumptions) == len(periods_one_year) == 12
    assert arrays_similar(consumption_joules,[245.8, 279.2, 291.5, 153.8, 108.2, 71.7, 192.0, 438.3, 390.2, 181.9, 159.7, 351.1])
