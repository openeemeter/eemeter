from eemeter.structures import ModelingPeriod
from datetime import datetime
import pytz
import pytest


@pytest.fixture(params=["BASELINE", "REPORTING"])
def interpretation(request):
    return request.param


def test_create(interpretation):
    start_date = datetime(2000, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2000, 1, 2, tzinfo=pytz.UTC)
    mp = ModelingPeriod(interpretation, start_date, end_date)
    assert mp.interpretation == interpretation
    assert mp.start_date == start_date
    assert mp.end_date == end_date


def test_bad_interprtation():
    interpretation = "INVALID"
    start_date = datetime(2000, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2000, 1, 2, tzinfo=pytz.UTC)
    with pytest.raises(ValueError):
        ModelingPeriod(interpretation, start_date, end_date)


def test_end_before_start(interpretation):
    start_date = datetime(2000, 1, 2, tzinfo=pytz.UTC)
    end_date = datetime(2000, 1, 1, tzinfo=pytz.UTC)
    with pytest.raises(ValueError):
        ModelingPeriod(interpretation, start_date, end_date)


def test_tz_unaware(interpretation):
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2000, 1, 2)
    with pytest.raises(ValueError):
        ModelingPeriod(interpretation, start_date, end_date)


def test_tz_both_dates_blank(interpretation):
    mp = ModelingPeriod(interpretation)
    assert mp.start_date is None
    assert mp.end_date is None


def test_tz_start_date_blank(interpretation):
    start_date = datetime(2000, 1, 2, tzinfo=pytz.UTC)
    mp = ModelingPeriod(interpretation, start_date=start_date)
    assert mp.start_date == start_date
    assert mp.end_date is None


def test_tz_end_date_blank(interpretation):
    end_date = datetime(2000, 1, 2, tzinfo=pytz.UTC)
    mp = ModelingPeriod(interpretation, end_date=end_date)
    assert mp.start_date is None
    assert mp.end_date == end_date


def test_repr(interpretation):
    mp = ModelingPeriod(interpretation)
    assert str(mp).startswith("ModelingPeriod")
