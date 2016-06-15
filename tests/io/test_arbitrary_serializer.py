from eemeter.io.serializers import ArbitrarySerializer
from datetime import datetime
import pandas as pd
import numpy as np
import pytz
import pytest


@pytest.fixture
def serializer():
    return ArbitrarySerializer()


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


def test_record_no_start(serializer):
    records = [
        {
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "value": 1,
        },
    ]
    with pytest.raises(ValueError):
        serializer.to_dataframe(records)


def test_record_no_end(serializer):
    records = [
        {
            "start": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "value": 1,
        },
    ]
    with pytest.raises(ValueError):
        serializer.to_dataframe(records)


def test_record_no_value(serializer):
    records = [
        {
            "start": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
        },
    ]
    with pytest.raises(ValueError):
        serializer.to_dataframe(records)


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


def test_multiple_records_no_gap(serializer):
    records = [
        {
            "start": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "value": 1,
        },
        {
            "start": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 3, tzinfo=pytz.UTC),
            "value": 2,
        },
    ]
    df = serializer.to_dataframe(records)
    assert df.value[datetime(2000, 1, 1, tzinfo=pytz.UTC)] == 1
    assert not df.estimated[datetime(2000, 1, 1, tzinfo=pytz.UTC)]

    assert df.value[datetime(2000, 1, 2, tzinfo=pytz.UTC)] == 2
    assert not df.estimated[datetime(2000, 1, 2, tzinfo=pytz.UTC)]

    assert pd.isnull(df.value[datetime(2000, 1, 3, tzinfo=pytz.UTC)])
    assert not df.estimated[datetime(2000, 1, 3, tzinfo=pytz.UTC)]


def test_multiple_records_with_gap(serializer):
    records = [
        {
            "start": datetime(2000, 1, 1, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 2, tzinfo=pytz.UTC),
            "value": 1,
        },
        {
            "start": datetime(2000, 1, 3, tzinfo=pytz.UTC),
            "end": datetime(2000, 1, 4, tzinfo=pytz.UTC),
            "value": 2,
        },
    ]
    df = serializer.to_dataframe(records)
    assert df.value[datetime(2000, 1, 1, tzinfo=pytz.UTC)] == 1
    assert not df.estimated[datetime(2000, 1, 1, tzinfo=pytz.UTC)]

    assert pd.isnull(df.value[datetime(2000, 1, 2, tzinfo=pytz.UTC)])
    assert not df.estimated[datetime(2000, 1, 2, tzinfo=pytz.UTC)]

    assert df.value[datetime(2000, 1, 3, tzinfo=pytz.UTC)] == 2
    assert not df.estimated[datetime(2000, 1, 3, tzinfo=pytz.UTC)]

    assert pd.isnull(df.value[datetime(2000, 1, 4, tzinfo=pytz.UTC)])
    assert not df.estimated[datetime(2000, 1, 4, tzinfo=pytz.UTC)]

def test_to_records(serializer):

    data = {"value": [1, np.nan], "estimated": [True, False]}
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=2, freq='D')
    df = pd.DataFrame(data, index=index, columns=columns)

    records = serializer.to_records(df)
    assert len(records) == 1
    assert records[0]["start"] == datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert records[0]["end"] == datetime(2000, 1, 2, tzinfo=pytz.UTC)
    assert records[0]["value"] == 1
    assert records[0]["estimated"]
