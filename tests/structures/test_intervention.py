from eemeter.structures import Intervention
import pytest
from datetime import datetime
import pytz


def test_naive_start_date():
    with pytest.raises(ValueError):
        intervention = Intervention(datetime(2000, 1, 1))


def test_naive_end_date():
    with pytest.raises(ValueError):
        intervention = Intervention(datetime(2000, 1, 1, tzinfo=pytz.UTC),
                                    datetime(2000, 1, 2))


def test_dates_wrong_order():
    with pytest.warns(UserWarning):
        intervention = Intervention(datetime(2000, 1, 2, tzinfo=pytz.UTC),
                                    datetime(2000, 1, 1, tzinfo=pytz.UTC))


def test_ok_start_only():
    intervention = Intervention(datetime(2000, 1, 1, tzinfo=pytz.UTC))
    assert intervention.start_date == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert intervention.end_date is None


def test_ok_start_and_end():
    intervention = Intervention(datetime(2000, 1, 1, tzinfo=pytz.UTC),
                                datetime(2000, 1, 2, tzinfo=pytz.UTC))
    assert intervention.start_date == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert intervention.end_date == datetime(2000, 1, 2, tzinfo=pytz.UTC)
