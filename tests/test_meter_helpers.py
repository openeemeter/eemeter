from eemeter.meter import RecentReadingMeter
from eemeter.meter import EstimatedReadingConsolidationMeter
from eemeter.meter import TimeSpanMeter
from eemeter.meter import TotalHDDMeter
from eemeter.meter import TotalCDDMeter
from eemeter.meter import MeetsThresholds
from eemeter.meter import NormalAnnualHDD
from eemeter.meter import NormalAnnualCDD
from eemeter.meter import NPeriodsMeetingHDDPerDayThreshold
from eemeter.meter import NPeriodsMeetingCDDPerDayThreshold


from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory

from eemeter.generator import ConsumptionGenerator
from eemeter.generator import generate_periods

from fixtures.consumption import time_span_1
from fixtures.consumption import generated_consumption_history_with_hdd_1
from fixtures.consumption import generated_consumption_history_with_cdd_1
from fixtures.consumption import generated_consumption_history_with_n_periods_hdd_1
from fixtures.consumption import generated_consumption_history_with_n_periods_cdd_1
from fixtures.consumption import time_span_1

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from datetime import datetime
from datetime import timedelta

from numpy.testing import assert_allclose

RTOL=1e-2
ATOL=1e-2

import pytest

def test_recent_reading_meter():
    recent_consumption = Consumption(0,"kWh","electricity",datetime.now() - timedelta(days=390),datetime.now() - timedelta(days=360))
    old_consumption = Consumption(0,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1))
    no_ch = ConsumptionHistory([])
    old_ch = ConsumptionHistory([old_consumption])
    recent_ch = ConsumptionHistory([recent_consumption])
    mixed_ch = ConsumptionHistory([recent_consumption,old_consumption])

    meter = RecentReadingMeter(n_days=365)
    assert not meter.evaluate(consumption_history=no_ch,fuel_type="electricity")["recent_reading"]
    assert not meter.evaluate(consumption_history=old_ch,fuel_type="electricity")["recent_reading"]
    assert meter.evaluate(consumption_history=recent_ch,fuel_type="electricity")["recent_reading"]
    assert meter.evaluate(consumption_history=mixed_ch,fuel_type="electricity")["recent_reading"]
    assert not meter.evaluate(consumption_history=mixed_ch,fuel_type="natural_gas")["recent_reading"]

    meter = RecentReadingMeter(n_days=365,since_date=datetime.now() + timedelta(days=1000))
    assert not meter.evaluate(consumption_history=mixed_ch,fuel_type="electricity")["recent_reading"]
    assert not meter.evaluate(consumption_history=mixed_ch,fuel_type="natural_gas")["recent_reading"]

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

def test_time_span_meter(time_span_1):
    ch, fuel_type, n_days = time_span_1
    meter = TimeSpanMeter()
    assert n_days == meter.evaluate(consumption_history=ch,fuel_type=fuel_type)["time_span"]

def test_total_hdd_meter(generated_consumption_history_with_hdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, hdd = generated_consumption_history_with_hdd_1
    meter = TotalHDDMeter(base=65,temperature_unit_str="degF")
    result = meter.evaluate(consumption_history=ch,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert hdd == result["total_hdd"]

def test_total_cdd_meter(generated_consumption_history_with_cdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, cdd = generated_consumption_history_with_cdd_1
    meter = TotalCDDMeter(base=65,temperature_unit_str="degF")
    result = meter.evaluate(consumption_history=ch,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert cdd == result["total_cdd"]

def test_meets_thresholds():
    meter = MeetsThresholds(values=["one","two","three","four"],
                            thresholds=[0,"two","four",3],
                            operations=["lt","lte","gt","gte"],
                            proportions=[1,1,.5,2],
                            output_names=["one_lt_zero","two_lte_two","three_gt_half_two","four_gte_twice_three"])
    result = meter.evaluate(one=1,two=2,three=3,four=4)
    assert not result["one_lt_zero"]
    assert result["two_lte_two"]
    assert result["three_gt_half_two"]
    assert not result["four_gte_twice_three"]

def test_normal_annual_hdd(tmy3_722880_weather_source):
    meter = NormalAnnualHDD(base=65,temperature_unit_str="degF")
    result = meter.evaluate(weather_normal_source=tmy3_722880_weather_source)
    assert_allclose(result["normal_annual_hdd"],1578.588175669573,rtol=RTOL,atol=ATOL)

def test_normal_annual_cdd(tmy3_722880_weather_source):
    meter = NormalAnnualCDD(base=65,temperature_unit_str="degF")
    result = meter.evaluate(weather_normal_source=tmy3_722880_weather_source)
    assert_allclose(result["normal_annual_cdd"],1248.4575607999941,rtol=RTOL,atol=ATOL)

def test_n_periods_meeting_hdd_per_day_threshold(generated_consumption_history_with_n_periods_hdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, n_periods_lt, n_periods_gt, hdd = generated_consumption_history_with_n_periods_hdd_1
    meter_lt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="lt")
    meter_gt = NPeriodsMeetingHDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="gt")
    result_lt = meter_lt.evaluate(consumption_history=ch,
                            hdd=hdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate(consumption_history=ch,
                            hdd=hdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert n_periods_lt == result_lt["n_periods"]
    assert n_periods_gt == result_gt["n_periods"]

def test_n_periods_meeting_cdd_per_day_threshold(generated_consumption_history_with_n_periods_cdd_1,gsod_722880_2012_2014_weather_source):
    ch, fuel_type, n_periods_lt, n_periods_gt, cdd = generated_consumption_history_with_n_periods_cdd_1
    meter_lt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="lt")
    meter_gt = NPeriodsMeetingCDDPerDayThreshold(base=65,temperature_unit_str="degF",operation="gt")
    result_lt = meter_lt.evaluate(consumption_history=ch,
                            cdd=cdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    result_gt = meter_gt.evaluate(consumption_history=ch,
                            cdd=cdd,
                            fuel_type=fuel_type,
                            weather_source=gsod_722880_2012_2014_weather_source)
    assert n_periods_lt == result_lt["n_periods"]
    assert n_periods_gt == result_gt["n_periods"]
