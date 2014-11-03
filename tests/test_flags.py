from eemeter.flags import BaseFlag
from eemeter.flags import NoneInTimeRangeFlag
from eemeter.flags import OverlappingPeriodsFlag
from eemeter.flags import MissingPeriodsFlag
from eemeter.flags import TooManyEstimatedPeriodsFlag
from eemeter.flags import ShortTimeSpanFlag
from eemeter.flags import InsufficientTemperatureRangeFlag
from eemeter.flags import MixedFuelTypeFlag

import pytest

##### Fixtures #####

@pytest.fixture("module")
def raised_base_flag():
    return BaseFlag(True)

@pytest.fixture("module")
def unraised_base_flag():
    return BaseFlag(False)

@pytest.fixture("module")
def none_in_time_range_flag():
    return NoneInTimeRangeFlag(False)

@pytest.fixture("module")
def overlapping_periods_flag():
    return OverlappingPeriodsFlag(False)

@pytest.fixture("module")
def missing_periods_flag():
    return MissingPeriodsFlag(False)

@pytest.fixture("module",params=[1,2])
def too_many_estimated_periods_flag(request):
    return TooManyEstimatedPeriodsFlag(False,request.param)

@pytest.fixture("module",params=[1,183])
def short_time_span_flag(request):
    return ShortTimeSpanFlag(False,request.param)

@pytest.fixture("module")
def insufficient_temperature_range_flag():
    return InsufficientTemperatureRangeFlag(False)

@pytest.fixture("module")
def mixed_fuel_type_flag():
    return MixedFuelTypeFlag(False)

##### Tests #####

def test_base_flag_raised(raised_base_flag):
    assert raised_base_flag.raised

def test_base_flag_unraised(unraised_base_flag):
    assert not unraised_base_flag.raised

def test_base_flag_has_none_description(raised_base_flag):
    assert raised_base_flag.description() == None

def test_none_in_time_range_flag(none_in_time_range_flag):
    assert not none_in_time_range_flag.raised
    assert "None in time range" == none_in_time_range_flag.description()

def test_overlapping_periods_flag(overlapping_periods_flag):
    assert not overlapping_periods_flag.raised
    assert "Overlapping time periods" == overlapping_periods_flag.description()

def test_missing_periods_flag(missing_periods_flag):
    assert not missing_periods_flag.raised
    assert "Missing time periods" == missing_periods_flag.description()

def test_too_many_estimated_periods_flag(too_many_estimated_periods_flag):
    assert not too_many_estimated_periods_flag.raised
    assert "More than {} estimated periods".format(too_many_estimated_periods_flag.limit) == \
            too_many_estimated_periods_flag.description()

def test_short_time_span_flag(short_time_span_flag):
    assert not short_time_span_flag.raised
    assert "Fewer than {} days in sample".format(short_time_span_flag.limit) == \
            short_time_span_flag.description()

def test_insufficient_temperature_range_flag(insufficient_temperature_range_flag):
    assert not insufficient_temperature_range_flag.raised
    assert "Insufficient temperature range" == insufficient_temperature_range_flag.description()

def test_mixed_fuel_type_flag(mixed_fuel_type_flag):
    assert not mixed_fuel_type_flag.raised
    assert "Mixed fuel types" == mixed_fuel_type_flag.description()
