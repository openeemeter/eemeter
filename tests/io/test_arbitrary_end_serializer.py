from eemeter.io.serializers import ArbitraryEndSerializer
from datetime import datetime
import pandas as pd
import pytz
import pytest


@pytest.fixture
def serializer():
    return ArbitraryEndSerializer()

def test_no_records(serializer):
    df = serializer.to_dataframe([])
    assert df.empty
    assert all(df.columns == ["value", "estimated"])

def test_single_valid_record(serializer):
    records = [
        {
            "start": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "value": 1,
        },
    ]
    df = serializer.to_dataframe(records)
    assert df.value[datetime(2000, 1, 1, tzinfo=pytz.UTC)] == 1
    assert not df.estimated[datetime(2000, 1, 1, tzinfo=pytz.UTC)]

    assert pd.isnull(df.value[datetime(2000, 1, 2, tzinfo=pytz.UTC)])
    assert not df.estimated[datetime(2000, 1, 2, tzinfo=pytz.UTC)]


def test_single_valid_record_end_only(serializer):
    records = [
        {
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "value": 1,
        },
    ]
    df = serializer.to_dataframe(records)
    assert pd.isnull(df.value[datetime(2000, 1, 2, tzinfo=pytz.UTC)])
    assert not df.estimated[datetime(2000, 1, 2, tzinfo=pytz.UTC)]


def test_single_valid_record_with_estimated(serializer):
    records = [
        {
            "start": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "value": 1,
            "estimated": True,
        },
    ]
    df = serializer.to_dataframe(records)
    assert df.value[datetime(2000, 1, 1, tzinfo=pytz.UTC)] == 1
    assert df.estimated[datetime(2000, 1, 1, tzinfo=pytz.UTC)]

    assert pd.isnull(df.value[datetime(2000, 1, 2, tzinfo=pytz.UTC)])
    assert not df.estimated[datetime(2000, 1, 2, tzinfo=pytz.UTC)]


def test_record_no_end(serializer):
    records = [
        {
            "value": 1,
        },
    ]
    with pytest.raises(ValueError):
        serializer.to_dataframe(records)


def test_record_no_value(serializer):
    records = [
        {
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
        },
    ]
    with pytest.raises(ValueError):
        serializer.to_dataframe(records)


def test_multiple_records(serializer):
    records = [
        {
            "end": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "value": 1,
        },
        {
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "value": 2,
        },
    ]
    df = serializer.to_dataframe(records)
    assert df.value[datetime(2000, 1, 1, tzinfo=pytz.UTC)] == 2
    assert not df.estimated[datetime(2000, 1, 1, tzinfo=pytz.UTC)]

    assert pd.isnull(df.value[datetime(2000, 1, 2, tzinfo=pytz.UTC)])
    assert not df.estimated[datetime(2000, 1, 2, tzinfo=pytz.UTC)]


def test_record_end_before_start(serializer):
    records = [
        {
            "start": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "value": 1,
        },
    ]
    with pytest.raises(ValueError):
        serializer.to_dataframe(records)
