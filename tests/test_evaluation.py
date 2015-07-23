from eemeter.evaluation import Period

from datetime import datetime
from datetime import timedelta

import pytest

def test_period_both_closed():
    p = Period(datetime(2014,1,1),datetime(2014,1,2))
    assert p.start == datetime(2014,1,1)
    assert p.end == datetime(2014,1,2)
    assert p.timedelta == timedelta(days=1)

def test_period_start_closed():
    p = Period(start=datetime(2014,1,1))
    assert p.start == datetime(2014,1,1)
    assert p.end is None
    assert p.timedelta is None

def test_period_end_closed():
    p = Period(end=datetime(2014,1,2))
    assert p.start is None
    assert p.end == datetime(2014,1,2)
    assert p.timedelta is None

def test_period_both_open():
    p = Period()
    assert p.start is None
    assert p.end is None
    assert p.timedelta is None
