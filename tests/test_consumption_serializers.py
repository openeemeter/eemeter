from datetime import datetime
from eemeter.consumption import (
    ArbitraryConsumptionSerializer,
    ArbitraryStartConsumptionSerializer,
)
import numpy as np
from numpy.testing import assert_allclose
import pytest
import pytz


@pytest.fixture
def arbitrary_serializer():
    return ArbitraryConsumptionSerializer()

@pytest.fixture
def arbitrary_start_serializer():
    return ArbitraryStartConsumptionSerializer()

def test_arbitrary_serializer_to_dataframe_empty(arbitrary_serializer):
    df = arbitrary_serializer.to_dataframe([])
    assert df.empty
    assert all(df.columns == ["value", "estimated"])

def test_arbitrary_serializer_to_dataframe_invalid_records(arbitrary_serializer):
    # missing field
    records = [{"start": None}]
    with pytest.raises(ValueError):
        df = arbitrary_serializer.to_dataframe(records)

    # not a datetime
    records = [{"start": None, "end": None, "value": None}]
    with pytest.raises(AttributeError):
        df = arbitrary_serializer.to_dataframe(records)

    # no timezone info
    records = [
        {
            "start": datetime(2011, 1, 1),
            "end": datetime(2011, 1, 2),
            "value": None,
        }
    ]
    with pytest.raises(ValueError):
        df = arbitrary_serializer.to_dataframe(records)

    # None value becomes NaN
    records = [
        {
            "start": datetime(2011, 1, 1, tzinfo=pytz.UTC),
            "end": datetime(2011, 1, 2, tzinfo=pytz.UTC),
            "value": None,
        }
    ]
    df = arbitrary_serializer.to_dataframe(records)
    assert df.value[datetime(2011, 1, 1, tzinfo=pytz.UTC)] is not None

def test_arbitrary_serializer_to_dataframe_valid_records(arbitrary_serializer):
    records = [
        {
            "start": datetime(2011, 1, 1, tzinfo=pytz.UTC),
            "end": datetime(2011, 1, 2, tzinfo=pytz.UTC),
            "value": 1,
        },
        {
            "start": datetime(2011, 1, 3, tzinfo=pytz.UTC),
            "end": datetime(2011, 1, 4, tzinfo=pytz.UTC),
            "value": 2,
        }
    ]
    df = arbitrary_serializer.to_dataframe(records)
    assert_allclose(df.value.values, [1., np.nan, 2., np.nan])
    assert not any(df.estimated.values)

def test_arbitrary_start_serializer_to_dataframe_empty(arbitrary_start_serializer):
    df = arbitrary_start_serializer.to_dataframe([])
    assert df.empty
    assert all(df.columns == ["value", "estimated"])

def test_arbitrary_start_serializer_to_dataframe_valid_records(arbitrary_start_serializer):
    records = [
        {
            "start": datetime(2011, 1, 1, tzinfo=pytz.UTC),
            "value": 1,
        },
        {
            "start": datetime(2011, 1, 3, tzinfo=pytz.UTC),
            "value": 2,
        }
    ]
    df = arbitrary_start_serializer.to_dataframe(records)
    assert_allclose(df.value.values, [1., np.nan])
    assert not any(df.estimated.values)

    records = [
        {
            "start": datetime(2011, 1, 1, tzinfo=pytz.UTC),
            "value": 1,
        },
        {
            "start": datetime(2011, 1, 3, tzinfo=pytz.UTC),
            "end": datetime(2011, 1, 4, tzinfo=pytz.UTC),
            "value": 2,
        }
    ]
    df = arbitrary_start_serializer.to_dataframe(records)
    assert_allclose(df.value.values, [1., 2., np.nan])
    assert not any(df.estimated.values)
