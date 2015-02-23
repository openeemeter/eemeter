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

def test_estimated_reading_consolidation_meter_single_fuel_type():
    ch1 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,3,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch2 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])


    ch3 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"kWh","electricity",datetime(2012,5,1),datetime(2012,6,1),estimated=True)
            ])

    ch4 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"kWh","electricity",datetime(2012,5,1),datetime(2012,6,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,6,1),datetime(2012,7,1),estimated=True)
            ])

    ch5 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,2,10),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,2,10),datetime(2012,3,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch6 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,1,10),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,1,10),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch7 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,1,10),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,1,10),datetime(2012,2,1))
            ])

    ch_no_est = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1))
            ])

    meter = EstimatedReadingConsolidationMeter()
    result1 = meter.evaluate(consumption_history=ch1)
    result2 = meter.evaluate(consumption_history=ch2)
    result3 = meter.evaluate(consumption_history=ch3)
    result4 = meter.evaluate(consumption_history=ch4)
    result5 = meter.evaluate(consumption_history=ch5)
    result6 = meter.evaluate(consumption_history=ch6)
    result7 = meter.evaluate(consumption_history=ch7)

    for result in [result1,result2,result3,result4,result5,result6,result7]:
        for c1,c2 in zip(result["consumption_history_no_estimated"].iteritems(),
                         ch_no_est.iteritems()):
            assert c1 == c2

def test_estimated_reading_consolidation_meter_multiple_fuel_type():
    ch1 = ConsumptionHistory([
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,3,1),estimated=True),
            Consumption(0,"kWh","electricity",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,2,1),datetime(2012,3,1),estimated=True),
            Consumption(0,"therm","natural_gas",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,4,1),datetime(2012,5,1))
            ])

    ch_no_est = ConsumptionHistory([
            Consumption(0,"therm","natural_gas",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(0,"kWh","electricity",datetime(2012,2,1),datetime(2012,4,1)),
            Consumption(0,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(0,"therm","natural_gas",datetime(2012,4,1),datetime(2012,5,1))
            ])

    meter = EstimatedReadingConsolidationMeter()
    result1 = meter.evaluate(consumption_history=ch1)

    for result in [result1]:
        for c1,c2 in zip(result["consumption_history_no_estimated"].get("electricity"),
                         ch_no_est.get("electricity")):
            assert c1 == c2
        for c1,c2 in zip(result["consumption_history_no_estimated"].get("natural_gas"),
                         ch_no_est.get("natural_gas")):
            assert c1 == c2

