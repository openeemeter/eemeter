import pytz
from datetime import datetime
import pytest

from eemeter.structures import ModelingPeriod
from eemeter.structures import ModelingPeriodSet


def test_create_blank():
    modeling_periods = {}
    groupings = []
    mps = ModelingPeriodSet(modeling_periods, groupings)

    groups = list(mps.iter_modeling_period_groups())
    assert len(groups) == 0

    modeling_periods = list(mps.iter_modeling_periods())
    assert len(modeling_periods) == 0


def test_create_basic():
    modeling_period_1 = ModelingPeriod(
        "BASELINE",
        end_date=datetime(2000, 1, 1, tzinfo=pytz.UTC),
    )
    modeling_period_2 = ModelingPeriod(
        "REPORTING",
        start_date=datetime(2000, 1, 1, tzinfo=pytz.UTC),
    )
    modeling_periods = {
        "modeling_period_1": modeling_period_1,
        "modeling_period_2": modeling_period_2,
    }

    grouping = [
        ("modeling_period_1", "modeling_period_2"),
    ]

    mps = ModelingPeriodSet(modeling_periods, grouping)

    groups = list(mps.iter_modeling_period_groups())
    assert len(groups) == 1
    group = groups[0]
    assert len(group) == 2
    assert len(group[0]) == 2
    assert group[0][0] == "modeling_period_1"
    assert group[0][1] == "modeling_period_2"
    assert group[1][0] == modeling_period_1
    assert group[1][1] == modeling_period_2

    modeling_periods = list(mps.iter_modeling_periods())
    assert len(modeling_periods) == 2
    assert modeling_periods[0][0] == "modeling_period_1"
    assert modeling_periods[1][0] == "modeling_period_2"
    assert modeling_periods[0][1] == modeling_period_1
    assert modeling_periods[1][1] == modeling_period_2


def test_grouping_out_of_order():
    modeling_periods = {
        "modeling_period_1": ModelingPeriod(
            "BASELINE",
            end_date=datetime(2000, 1, 1, tzinfo=pytz.UTC),
        ),
        "modeling_period_2": ModelingPeriod(
            "REPORTING",
            start_date=datetime(2000, 2, 1, tzinfo=pytz.UTC),
        ),
    }

    grouping = [
        ("modeling_period_2", "modeling_period_1"),
    ]

    with pytest.raises(ValueError):
        ModelingPeriodSet(modeling_periods, grouping)


def test_grouping_misreference():
    modeling_periods = {}

    grouping = [
        ("typo1", "typo2"),
    ]

    with pytest.raises(ValueError):
        ModelingPeriodSet(modeling_periods, grouping)


def test_grouping_wrong_length():
    modeling_periods = {
        "modeling_period_1": ModelingPeriod(
            "BASELINE",
            end_date=datetime(2000, 1, 1, tzinfo=pytz.UTC),
        ),
        "modeling_period_2": ModelingPeriod(
            "REPORTING",
            start_date=datetime(2000, 2, 1, tzinfo=pytz.UTC),
        ),
    }

    grouping = [
        ("modeling_period1", )
    ]

    with pytest.raises(ValueError):
        ModelingPeriodSet(modeling_periods, grouping)


def test_repr():

    modeling_period_1 = ModelingPeriod(
        "BASELINE",
        end_date=datetime(2000, 1, 1, tzinfo=pytz.UTC),
    )
    modeling_period_2 = ModelingPeriod(
        "REPORTING",
        start_date=datetime(2000, 1, 1, tzinfo=pytz.UTC),
    )
    modeling_periods = {
        "modeling_period_1": modeling_period_1,
        "modeling_period_2": modeling_period_2,
    }

    grouping = [
        ("modeling_period_1", "modeling_period_2"),
    ]

    mps = ModelingPeriodSet(modeling_periods, grouping)

    assert str(mps).startswith('ModelingPeriodSet(modeling_periods={')
    assert str(mps).endswith(
        ', groupings=[(\'modeling_period_1\', \'modeling_period_2\')])')
