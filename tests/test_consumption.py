from eemeter.consumption import ConsumptionData
from datetime import timedelta
from datetime import datetime
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import warnings

from eemeter.consumption import Consumption
from eemeter.consumption import DatetimePeriod
from eemeter.consumption import DateRangeException
from eemeter.consumption import ConsumptionHistory

from pint.unit import UndefinedUnitError

import pytest

EPSILON = 1e-6
RTOL = 1e-3
ATOL = 1e-3

@pytest.fixture(params=["electricity",
                        "natural_gas"])
def fuel_type(request):
    return request.param

@pytest.fixture(params=["kWh",
                        "therm"])
def unit_name(request):
    return request.param

@pytest.fixture(params=["arbitrary",
                        "billing"])
def record_type_arbitrary(request):
    return request.param

@pytest.fixture(params=["arbitrary_start",
                        "billing_start"])
def record_type_arbitrary_start(request):
    return request.param

@pytest.fixture(params=["arbitrary_end",
                        "billing_end"])
def record_type_arbitrary_end(request):
    return request.param

@pytest.fixture(params=[
        [{"start": datetime(2015,1,5,0,0,0), "value": 5},
         {"start": datetime(2015,1,4,0,0,0), "value": 4},
         {"start": datetime(2015,1,3,0,0,0), "value": 3},
         {"start": datetime(2015,1,2,0,0,0), "value": 2},
         {"start": datetime(2015,1,1,0,0,0), "value": 1, "estimated": True}],
        [{"start": datetime(2015,1,1,0,0,0), "value": 1, "estimated": True},
         {"start": datetime(2015,1,2,0,0,0), "value": 2},
         {"start": datetime(2015,1,3,0,0,0), "value": 3},
         {"start": datetime(2015,1,4,0,0,0), "value": 4},
         {"start": datetime(2015,1,5,0,0,0), "value": 5, "estimated": False}],
        [{"start": datetime(2015,1,2,0,0,0), "value": 2},
            {"start": datetime(2015,1,1,0,0,0), "value": 1, "estimated": True},
         {"start": datetime(2015,1,4,0,0,0), "value": 4},
         {"start": datetime(2015,1,5,0,0,0), "value": 5},
         {"start": datetime(2015,1,3,0,0,0), "value": 3}],
        ])
def records_interval_start_daily_all(request):
    return request.param

@pytest.fixture(params=[
        [{"end": datetime(2015,1,5,0,0,0), "value": 5},
         {"end": datetime(2015,1,4,0,0,0), "value": 4},
         {"end": datetime(2015,1,3,0,0,0), "value": 3},
         {"end": datetime(2015,1,2,0,0,0), "value": 2},
         {"end": datetime(2015,1,1,0,0,0), "value": 1}],
        [{"end": datetime(2015,1,1,0,0,0), "value": 1},
         {"end": datetime(2015,1,2,0,0,0), "value": 2},
         {"end": datetime(2015,1,3,0,0,0), "value": 3},
         {"end": datetime(2015,1,4,0,0,0), "value": 4},
         {"end": datetime(2015,1,5,0,0,0), "value": 5}],
        [{"end": datetime(2015,1,2,0,0,0), "value": 2},
         {"end": datetime(2015,1,1,0,0,0), "value": 1},
         {"end": datetime(2015,1,4,0,0,0), "value": 4},
         {"end": datetime(2015,1,5,0,0,0), "value": 5},
         {"end": datetime(2015,1,3,0,0,0), "value": 3}],
        ])
def records_interval_end_daily_all(request):
    return request.param

@pytest.fixture
def records_interval_start_daily_missing_date(request):
    records = [
        {"start": datetime(2015,1,1,0,0,0), "value": 1},
        {"start": datetime(2015,1,2,0,0,0), "value": 2},
        {"start": datetime(2015,1,4,0,0,0), "value": 4},
        {"start": datetime(2015,1,5,0,0,0), "value": 5}]
    return records

@pytest.fixture
def records_interval_start_daily_misaligned_date(request):
    records = [
        {"start": datetime(2015,1,1,0,0,0), "value": 1},
        {"start": datetime(2015,1,2,0,0,0), "value": 2},
        {"start": datetime(2015,1,3,0,0,1), "value": 3},
        {"start": datetime(2015,1,4,0,0,0), "value": 4},
        {"start": datetime(2015,1,5,0,0,0), "value": 5}]
    return records

@pytest.fixture
def records_interval_start_daily_overlapping_date(request):
    records = [
        {"start": datetime(2015,1,1,0,0,0), "value": 1},
        {"start": datetime(2015,1,2,0,0,0), "value": 2},
        {"start": datetime(2015,1,3,0,0,0), "value": 3},
        {"start": datetime(2015,1,3,0,0,0), "value": 3},
        {"start": datetime(2015,1,4,0,0,0), "value": 4},
        {"start": datetime(2015,1,5,0,0,0), "value": 5}]
    return records

@pytest.fixture
def records_interval_start_daily_missing_value(request):
    records = [
        {"start": datetime(2015,1,1,0,0,0), "value": 1},
        {"start": datetime(2015,1,2,0,0,0), "value": 2},
        {"start": datetime(2015,1,3,0,0,0)},
        {"start": datetime(2015,1,4,0,0,0), "value": 4},
        {"start": datetime(2015,1,5,0,0,0), "value": 5}]
    return records

@pytest.fixture
def records_interval_start_daily_missing_start_key(request):
    records = [
        {"start": datetime(2015,1,1,0,0,0), "value": 1},
        {"end": datetime(2015,1,2,0,0,0), "value": 2},
        {"start": datetime(2015,1,3,0,0,0), "value": 3},
        {"start": datetime(2015,1,4,0,0,0), "value": 4},
        {"start": datetime(2015,1,5,0,0,0), "value": 5}]
    return records

@pytest.fixture
def records_interval_start_15min(request):
    records = [
        {"start": datetime(2015,1,1,0,0,0), "value": 1},
        {"start": datetime(2015,1,1,0,15,0), "value": 2},
        {"start": datetime(2015,1,1,0,30,0), "value": 3},
        {"start": datetime(2015,1,1,0,45,0), "value": 4},
        {"start": datetime(2015,1,1,1,0,0), "value": 5},
        {"start": datetime(2015,1,1,1,15,0), "value": 6}]
    return records

@pytest.fixture
def records_arbitrary_basic(request):
    records = [
        {"start": datetime(2015,2,1,0,0,0), "end": datetime(2015,2,28), "value": 0},
        {"start": datetime(2015,1,2,0,0,0), "end": datetime(2015,2,1), "value": 0},
        {"start": datetime(2015,1,1,1,4,5), "end": datetime(2015,1,1,2,4,5), "value": 1},
        {"start": datetime(2015,1,1,0,4,5), "end": datetime(2015,1,1,1,4,5), "value": np.nan},
        {"start": datetime(2015,1,1,0,0,0), "end": datetime(2015,1,1,0,4,5), }]
    return records

@pytest.fixture
def records_arbitrary_overlap(request):
    records = [
        {"start": datetime(2015,1,20,0,0,0), "end": datetime(2015,2,28), "value": 0},
        {"start": datetime(2015,1,2,0,0,0), "end": datetime(2015,2,1), "value": 0},
        {"start": datetime(2015,1,1,1,4,5), "end": datetime(2015,1,1,2,4,5), "value": 1},
        {"start": datetime(2015,1,1,0,4,5), "end": datetime(2015,1,1,1,4,5), "value": np.nan},
        {"start": datetime(2015,1,1,0,0,0), "end": datetime(2015,1,1,0,4,5), }]
    return records

@pytest.fixture(params=[
        [{"start": datetime(2015,2,1),"value":0},
         {"start": datetime(2015,1,2,0,0,0), "value":1},
         {"start": datetime(2015,1,1,1,4,5), "value":0},
         {"start": datetime(2015,1,1,0,4,5), "value":0},
         {"start": datetime(2015,1,1,0,0,0), "value":0}],
        [{"start": datetime(2015,1,2,0,0,0), "end": datetime(2015,2,1), "value":1},
         {"start": datetime(2015,1,1,1,4,5), "value":0},
         {"start": datetime(2015,1,1,0,4,5), "value":0},
         {"start": datetime(2015,1,1,0,0,0), "value":0}],
        ])
def records_arbitrary_start(request):
    return request.param

@pytest.fixture(params=[
        [{"end": datetime(2015,2,1), "value":0},
         {"end": datetime(2015,1,2,0,0,0), "value":1},
         {"end": datetime(2015,1,1,1,4,5), "value":0},
         {"end": datetime(2015,1,1,0,4,5), "value":0},
         {"end": datetime(2015,1,1,0,0,0),  "start": datetime(2014,2,2), "value":0}],
        [{"end": datetime(2015,2,1), "value":0},
         {"end": datetime(2015,1,2,0,0,0), "value":1},
         {"end": datetime(2015,1,1,1,4,5), "value":0},
         {"end": datetime(2015,1,1,0,4,5), "value":0},
         {"end": datetime(2015,1,1,0,0,0),  "value":0},
         {"end": datetime(2014,2,2,0,0,0),  "value":0}]
        ])
def records_arbitrary_end(request):
    return request.param

@pytest.fixture
def records_pulse(request):
    records = [
        {"pulse": datetime(2015,2,1)},
        {"pulse": datetime(2015,1,2,0,0,0)},
        {"pulse": datetime(2015,1,1,1,4,5)},
        {"pulse": datetime(2015,1,1,0,4,5)},
        {"pulse": datetime(2015,1,1,0,0,0)}]
    return records

##### tests #####

def test_consumption_data_interval_start_daily_all_has_correct_attributes(
        records_interval_end_daily_all, fuel_type, unit_name):
    cd = ConsumptionData(records_interval_end_daily_all, fuel_type, unit_name,
            freq="D")
    assert cd.fuel_type == fuel_type
    assert cd.unit_name == unit_name
    assert cd.freq == "D"
    assert cd.freq_timedelta == timedelta(days=1)
    assert type(cd.data.index) == pd.DatetimeIndex
    assert cd.data.index.shape == (5,)
    assert cd.data.values.shape == (5,)

def test_consumption_data_interval_start_daily_all_invalid_fuel_type(
        records_interval_end_daily_all, unit_name):
    with pytest.raises(ValueError):
        cd = ConsumptionData(records_interval_end_daily_all, "invalid", unit_name,
                freq="D")

def test_consumption_data_interval_start_daily_all_invalid_unit_name(
        records_interval_end_daily_all, fuel_type):
    with pytest.raises(ValueError):
        cd = ConsumptionData(records_interval_end_daily_all, fuel_type, "invalid",
                freq="D")

def test_consumption_data_interval_start_daily_all_freq_D(
        records_interval_start_daily_all, fuel_type, unit_name):
    cd = ConsumptionData(records_interval_start_daily_all,
            fuel_type, unit_name, freq="D")
    assert cd.freq_timedelta == timedelta(days=1)
    assert_allclose(cd.data.values,[1,2,3,4,5], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)
    assert_allclose(cd.estimated,[True,False,False,False,False], rtol=RTOL, atol=ATOL)

def test_consumption_data_interval_end_daily_all_freq_D(
        records_interval_end_daily_all, fuel_type, unit_name):
    cd = ConsumptionData(records_interval_end_daily_all,
            fuel_type, unit_name, freq="D")
    assert cd.freq_timedelta == timedelta(days=1)
    assert_allclose(cd.data.values,[1,2,3,4,5], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2014,12,31)

def test_consumption_data_interval_end_daily_all_freq_2D(
        records_interval_end_daily_all, fuel_type, unit_name):
    cd = ConsumptionData(records_interval_end_daily_all,
            fuel_type, unit_name, freq="2D")
    assert cd.freq_timedelta == timedelta(days=2)
    assert_allclose(cd.data.values,[1,3,5], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2014,12,30)

def test_consumption_data_interval_end_daily_all_freq_12H(
        records_interval_end_daily_all, fuel_type, unit_name):
    cd = ConsumptionData(records_interval_end_daily_all,
            fuel_type, unit_name, freq="12H")
    assert cd.freq_timedelta == timedelta(seconds=60*60*12)
    assert_allclose(cd.data.values,[1,np.nan,2,np.nan,3,np.nan,4,np.nan,5],
            rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2014,12,31,12)

def test_consumption_data_interval_start_daily_missing_start_key_freq_D(
        records_interval_start_daily_missing_start_key, fuel_type, unit_name):
    with pytest.raises(ValueError):
        cd = ConsumptionData(records_interval_start_daily_missing_start_key,
                fuel_type, unit_name, freq="D")

def test_consumption_data_interval_start_daily_missing_date_freq_D(
        records_interval_start_daily_missing_date, fuel_type, unit_name):
    cd = ConsumptionData(records_interval_start_daily_missing_date,
            fuel_type, unit_name, freq="D")
    assert cd.freq_timedelta == timedelta(days=1)
    assert_allclose(cd.data.values,[1,2,np.nan,4,5], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)

def test_consumption_data_interval_start_daily_misaligned_date_freq_D(
        records_interval_start_daily_misaligned_date, recwarn):
    cd = ConsumptionData(records_interval_start_daily_misaligned_date,
            "electricity", "kWh", freq="D")
    assert cd.freq_timedelta == timedelta(days=1)
    assert_allclose(cd.data.values,[1,2,np.nan,4,5], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)
    w = recwarn.pop(UserWarning)
    assert issubclass(w.category, UserWarning)

def test_consumption_data_interval_start_daily_overlapping_date_freq_D(
        records_interval_start_daily_overlapping_date, recwarn):
    cd = ConsumptionData(records_interval_start_daily_overlapping_date,
            "electricity", "kWh", freq="D")
    assert cd.freq_timedelta == timedelta(days=1)
    assert_allclose(cd.data.values,[1,2,3,4,5], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)
    w = recwarn.pop(UserWarning)
    assert issubclass(w.category, UserWarning)

def test_consumption_data_interval_start_daily_missing_value_freq_D(
        records_interval_start_daily_missing_value):
    cd = ConsumptionData(records_interval_start_daily_missing_value,
            "electricity", "kWh", freq="D")
    assert cd.freq_timedelta == timedelta(days=1)
    assert_allclose(cd.data.values,[1,2,np.nan,4,5], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)

def test_consumption_data_interval_start_15min(
        records_interval_start_15min, fuel_type, unit_name):
    cd = ConsumptionData(records_interval_start_15min,
            fuel_type, unit_name, freq="15T")
    assert cd.freq_timedelta == timedelta(seconds=60*15)
    assert_allclose(cd.data.values,[1,2,3,4,5,6], rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)

def test_consumption_data_arbitrary_basic(records_arbitrary_basic,
        fuel_type, unit_name, record_type_arbitrary):
    cd = ConsumptionData(records_arbitrary_basic,
            fuel_type, unit_name, record_type=record_type_arbitrary)
    with pytest.raises(AttributeError):
        assert cd.freq
    with pytest.raises(AttributeError):
        assert cd.freq_timedelta
    assert_allclose(cd.data.values,[np.nan,np.nan,1,np.nan,0,0,np.nan],
            rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)

def test_consumtion_data_arbitrary_overlap(records_arbitrary_overlap, recwarn):
    cd = ConsumptionData(records_arbitrary_overlap,
            "electricity", "kWh", record_type="arbitrary")
    assert_allclose(cd.data.values,[np.nan,np.nan,1,np.nan,0,np.nan],
            rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)
    assert cd.data.index[5] == datetime(2015,2,1)
    w = recwarn.pop(UserWarning)
    assert issubclass(w.category, UserWarning)

def test_consumption_data_arbitrary_start(records_arbitrary_start,
        fuel_type, unit_name, record_type_arbitrary_start):
    cd = ConsumptionData(records_arbitrary_start,
            fuel_type, unit_name, record_type=record_type_arbitrary_start)
    assert_allclose(cd.data.values,[0,0,0,1,np.nan],
            rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)
    assert cd.data.index[4] == datetime(2015,2,1)

def test_consumption_data_arbitrary_end(records_arbitrary_end,
        fuel_type, unit_name, record_type_arbitrary_end):
    cd = ConsumptionData(records_arbitrary_end,
            fuel_type, unit_name, record_type=record_type_arbitrary_end)
    assert_allclose(cd.data.values,[0,0,0,1,0,np.nan],
            rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2014,2,2)
    assert cd.data.index[5] == datetime(2015,2,1)

def test_consumption_data_pulse(records_pulse,
        fuel_type, unit_name):
    with pytest.raises(ValueError):
        cd = ConsumptionData(records_pulse,
                fuel_type, unit_name, record_type="pulse")
    with pytest.raises(ValueError):
        cd = ConsumptionData(records_pulse,
                fuel_type, unit_name, record_type="pulse", pulse_value=0)
    with pytest.raises(ValueError):
        cd = ConsumptionData(records_pulse,
                fuel_type, unit_name, record_type="pulse", pulse_value=-1)
    cd = ConsumptionData(records_pulse,
            fuel_type, unit_name, record_type="pulse", pulse_value=1)
    assert cd.pulse_value == 1
    assert_allclose(cd.data.values,[np.nan,1,1,1,1],
            rtol=RTOL, atol=ATOL)
    assert cd.data.index[0] == datetime(2015,1,1)
    assert cd.data.index[4] == datetime(2015,2,1)

def test_consumption_empty_(fuel_type, unit_name):
    cd = ConsumptionData([], fuel_type, unit_name, freq="S")
    assert_allclose(cd.data.values, [], rtol=RTOL, atol=ATOL)
    cd = ConsumptionData([], fuel_type, unit_name, record_type="arbitrary")
    assert_allclose(cd.data.values, [], rtol=RTOL, atol=ATOL)
    cd = ConsumptionData([], fuel_type, unit_name, record_type="arbitrary_start")
    assert_allclose(cd.data.values, [], rtol=RTOL, atol=ATOL)
    cd = ConsumptionData([], fuel_type, unit_name, record_type="arbitrary_end")
    assert_allclose(cd.data.values, [], rtol=RTOL, atol=ATOL)
    cd = ConsumptionData([], fuel_type, unit_name, record_type="pulse", pulse_value=1)
    assert_allclose(cd.data.values, [], rtol=RTOL, atol=ATOL)



@pytest.fixture(scope="module",
                params=[(0,"kWh","electricity",datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"kWh","electricity",datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"therms","electricity",datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"Btu","electricity",datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"Btu","natural_gas",datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"therms","natural_gas",datetime(2000,1,1),datetime(2000,1,31),False),
                        (0.0,"Btu","natural_gas",datetime(2000,1,1),datetime(2000,1,31),False),
                        (0,"kWh","electricity",datetime(2000,1,1),datetime(2000,1,31))])
def consumption_zero_one_month(request):
    return Consumption(*request.param)

@pytest.fixture
def consumption_list_one_year_electricity():
    return [Consumption(1000,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(1100,"kWh","electricity",datetime(2012,2,1),datetime(2012,3,1)),
            Consumption(1200,"kWh","electricity",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(1300,"kWh","electricity",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(1400,"kWh","electricity",datetime(2012,5,1),datetime(2012,6,1)),
            Consumption(1500,"kWh","electricity",datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(1400,"kWh","electricity",datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(1300,"kWh","electricity",datetime(2012,8,1),datetime(2012,9,1)),
            Consumption(1200,"kWh","electricity",datetime(2012,9,1),datetime(2012,10,1)),
            Consumption(1100,"kWh","electricity",datetime(2012,10,1),datetime(2012,11,1)),
            Consumption(1000,"kWh","electricity",datetime(2012,11,1),datetime(2012,12,1)),
            Consumption(900,"kWh","electricity",datetime(2012,12,1),datetime(2013,1,1))]

@pytest.fixture
def consumption_list_one_year_gas():
    return [Consumption(900,"thm","natural_gas",datetime(2012,1,1),datetime(2012,2,1)),
            Consumption(950,"thm","natural_gas",datetime(2012,2,1),datetime(2012,3,1)),
            Consumption(800,"kWh","natural_gas",datetime(2012,3,1),datetime(2012,4,1)),
            Consumption(700,"kWh","natural_gas",datetime(2012,4,1),datetime(2012,5,1)),
            Consumption(500,"kWh","natural_gas",datetime(2012,5,1),datetime(2012,6,1)),
            Consumption(100,"kWh","natural_gas",datetime(2012,6,1),datetime(2012,7,1)),
            Consumption(100,"kWh","natural_gas",datetime(2012,7,1),datetime(2012,8,1)),
            Consumption(100,"kWh","natural_gas",datetime(2012,8,1),datetime(2012,9,1)),
            Consumption(200,"kWh","natural_gas",datetime(2012,9,1),datetime(2012,10,1)),
            Consumption(400,"kWh","natural_gas",datetime(2012,10,1),datetime(2012,11,1)),
            Consumption(700,"kWh","natural_gas",datetime(2012,11,1),datetime(2012,12,1)),
            Consumption(900,"kWh","natural_gas",datetime(2012,12,1),datetime(2013,1,1))]

@pytest.fixture
def single_electricity_consumption():
    return Consumption(1000,"kWh","electricity",datetime(2012,1,1),datetime(2012,2,1))

##### Test cases #####

def test_datetime_period():
    dtp = DatetimePeriod(datetime(2000,1,1),datetime(2000,2,1))
    assert dtp.start == datetime(2000,1,1)
    assert dtp.end == datetime(2000,2,1)
    assert dtp.timedelta.days == 31
    assert str(dtp) == "DatetimePeriod({},{})".format(dtp.start,dtp.end)


def test_consumption_has_correct_attributes(consumption_zero_one_month):
    assert consumption_zero_one_month.joules == 0
    assert isinstance(consumption_zero_one_month.fuel_type,str)
    assert consumption_zero_one_month.start == datetime(2000,1,1)
    assert consumption_zero_one_month.end == datetime(2000,1,31)
    assert consumption_zero_one_month.estimated == False

def test_automatic_unit_conversion():
    btu_consumption = Consumption(1,"Btu","electricity",datetime(2000,1,1),datetime(2000,1,31),False)
    kwh_consumption = Consumption(1,"kWh","electricity",datetime(2000,1,1),datetime(2000,1,31),False)
    therm_consumption = Consumption(1,"therm","electricity",datetime(2000,1,1),datetime(2000,1,31),False)
    assert abs(1 - btu_consumption.btu) < EPSILON
    assert abs(1 - btu_consumption.BTU) < EPSILON
    assert abs(1 - btu_consumption.BTUs) < EPSILON
    assert abs(1 - btu_consumption.to("btu")) < EPSILON
    assert abs(1 - kwh_consumption.kWh) < EPSILON
    assert abs(1 - kwh_consumption.kilowatthour) < EPSILON
    assert abs(1 - kwh_consumption.kilowatthours) < EPSILON
    assert abs(1 - therm_consumption.thm) < EPSILON
    assert abs(1 - therm_consumption.therm) < EPSILON
    assert abs(1 - therm_consumption.therms) < EPSILON
    assert abs(therm_consumption.joules - btu_consumption.joules * 100000) < EPSILON
    with pytest.raises(UndefinedUnitError):
        assert abs(1 - kwh_consumption.kwh) < EPSILON

def test_feasible_consumption_start_end():
    with pytest.raises(DateRangeException):
        Consumption(1,"Btu","electricity",datetime(2000,1,2),datetime(2000,1,1))

def test_timedelta(consumption_zero_one_month):
    delta = consumption_zero_one_month.timedelta
    assert delta.days == 30

def test_consumption_has_string_representation(consumption_zero_one_month):
    assert "Consumption" in consumption_zero_one_month.__repr__()
    assert "Consumption" in str(consumption_zero_one_month)

def test_consumption_usage_per_day(single_electricity_consumption):
    assert abs(single_electricity_consumption.average_daily_usage("kWh") - 1000/31.) < EPSILON

def test_consumption_equality():
    c1 = Consumption(1,"kWh","electricity",datetime(2012,1,1),datetime(2013,1,1),estimated=True)
    c2 = Consumption(1,"kWh","electricity",datetime(2012,1,1),datetime(2013,1,1),estimated=True)
    c3 = Consumption(2,"kWh","electricity",datetime(2012,1,1),datetime(2013,1,1),estimated=True)
    c4 = Consumption(1,"therms","electricity",datetime(2012,1,1),datetime(2013,1,1),estimated=True)
    c5 = Consumption(1,"kWh","natural_gas",datetime(2012,1,1),datetime(2013,1,1),estimated=True)
    c6 = Consumption(1,"kWh","electricity",datetime(2012,1,2),datetime(2013,1,1),estimated=True)
    c7 = Consumption(1,"kWh","electricity",datetime(2012,1,1),datetime(2013,2,1),estimated=True)
    c8 = Consumption(1,"kWh","electricity",datetime(2012,1,1),datetime(2013,1,1),estimated=False)
    assert c1 == c2
    assert not c1 == c3
    assert not c1 == c4
    assert not c1 == c5
    assert not c1 == c6
    assert not c1 == c7
    assert not c1 == c8

def test_consumption_history(consumption_list_one_year_electricity,
                             consumption_list_one_year_gas):
    ch_elec = ConsumptionHistory(consumption_list_one_year_electricity)
    ch_gas = ConsumptionHistory(consumption_list_one_year_gas)

    # different ways to get the same data
    assert len(ch_elec.electricity) == 12
    assert len(ch_elec["electricity"]) == 12
    assert len(ch_elec.get("electricity")) == 12

    # other cases
    with pytest.raises(KeyError):
        assert len(ch_elec.natural_gas) == 0
        assert len(ch_gas.electricity) == 0
    assert len(ch_gas.natural_gas) == 12

    for consumption in ch_elec.electricity:
        assert consumption.kWh >= 0
    for consumption in ch_gas.natural_gas:
        assert consumption.therm >= 0

    assert ch_elec
    assert ch_gas

    for fuel_type,consumptions in ch_elec.fuel_types():
        consumptions.sort()

    assert len(ch_gas.before(datetime(2012,7,1)).natural_gas) == 6
    assert len(ch_gas.after(datetime(2012,6,30)).natural_gas) == 6
    assert len(ch_gas.before(datetime(2013,6,30)).natural_gas) == 12
    with pytest.raises(KeyError):
        assert ch_gas.before(datetime(2011,7,1)).natural_gas
        assert ch_gas.after(datetime(2013,6,30)).natural_gas

    assert ch_elec.get("nonexistent") == []

def test_consumption_average_daily_usage_no_division_by_zero():
    c = Consumption(10,"kWh","electricity",datetime(2012,1,1),datetime(2012,1,1))
    assert np.isnan(c.average_daily_usage("kWh"))
