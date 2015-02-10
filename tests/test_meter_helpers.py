from eemeter.meter import RecentReadingMeter
from eemeter.meter import EstimatedReadingConsolidationMeter

from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory

from datetime import datetime
from datetime import timedelta

import pytest

def test_recent_reading_meter():
    recent_consumption = Consumption(0,"kWh","electricity",datetime.now() - timedelta(days=390),datetime.now() - timedelta(days=360))
    old_consumption = Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1))
    no_ch = ConsumptionHistory([])
    old_ch = ConsumptionHistory([old_consumption])
    recent_ch = ConsumptionHistory([recent_consumption])
    mixed_ch = ConsumptionHistory([recent_consumption,old_consumption])

    meter = RecentReadingMeter(n_days=365)
    assert not meter.evaluate(consumption_history=no_ch)["recent_reading"]
    assert not meter.evaluate(consumption_history=old_ch)["recent_reading"]
    assert meter.evaluate(consumption_history=recent_ch)["recent_reading"]
    assert meter.evaluate(consumption_history=mixed_ch)["recent_reading"]

def test_estimated_reading_consolidation_meter():
    assert False

