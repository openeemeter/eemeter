from datetime import datetime
import pytz

import pytest
import numpy as np
import pandas as pd

from eemeter.processors.dispatchers import get_energy_modeling_dispatches
from eemeter.processors.collector import LogCollector
from eemeter.structures import (
    ModelingPeriod,
    ModelingPeriodSet,
    EnergyTrace,
    EnergyTraceSet,
)
from eemeter.modeling.formatters import ModelDataFormatter
from eemeter.modeling.models import SeasonalElasticNetCVModel


@pytest.fixture
def modeling_period_set():
    modeling_period_1 = ModelingPeriod(
        "BASELINE",
        end_date=datetime(2000, 1, 3, tzinfo=pytz.UTC),
    )
    modeling_period_2 = ModelingPeriod(
        "REPORTING",
        start_date=datetime(2000, 1, 3, tzinfo=pytz.UTC),
    )
    modeling_periods = {
        "modeling_period_1": modeling_period_1,
        "modeling_period_2": modeling_period_2,
    }

    grouping = [
        ("modeling_period_1", "modeling_period_2"),
    ]

    return ModelingPeriodSet(modeling_periods, grouping)


@pytest.fixture
def trace_set():
    columns = {
        "value": [1, 1, 1, 1, np.nan],
        "estimated": [False, False, False, False, False]
    }
    column_names = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=5, freq='D')
    data = pd.DataFrame(columns, index=index, columns=column_names)

    trace = EnergyTrace("ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED", data=data,
                        unit="KWH")

    return EnergyTraceSet([trace], ["trace"])


@pytest.fixture
def placeholder_trace_set():
    trace = EnergyTrace("ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED",
                        placeholder=True)

    return EnergyTraceSet([trace], ["trace"])


def test_basic_usage(modeling_period_set, trace_set):
    lc = LogCollector()

    with lc.collect_logs("get_energy_modeling_dispatches") as logger:
        dispatches = get_energy_modeling_dispatches(
                logger, modeling_period_set, trace_set)

    assert len(dispatches) == 2
    dispatch1 = dispatches[("modeling_period_1", "trace")]
    assert isinstance(dispatch1["formatter"], ModelDataFormatter)
    assert isinstance(dispatch1["model"], SeasonalElasticNetCVModel)
    assert isinstance(dispatch1["filtered_trace"], EnergyTrace)

    dispatch2 = dispatches[("modeling_period_2", "trace")]
    assert isinstance(dispatch2["formatter"], ModelDataFormatter)
    assert isinstance(dispatch2["model"], SeasonalElasticNetCVModel)
    assert isinstance(dispatch2["filtered_trace"], EnergyTrace)

    logs = lc.items["get_energy_modeling_dispatches"]
    print(logs)
    assert "INFO - Determined frequency of 'D' for EnergyTrace 'trace'." \
        in logs[0]
    assert "INFO - Successfully dispatched formatter" \
        in logs[1]
    assert "INFO - Determined frequency of 'D' for EnergyTrace 'trace'." \
        in logs[2]


def test_placeholder_trace(modeling_period_set, placeholder_trace_set):
    lc = LogCollector()

    with lc.collect_logs("get_energy_modeling_dispatches") as logger:
        dispatches = get_energy_modeling_dispatches(
                logger, modeling_period_set, placeholder_trace_set)

    assert len(dispatches) == 2
    dispatch1 = dispatches[("modeling_period_1", "trace")]
    assert dispatch1["formatter"] is None
    assert dispatch1["model"] is None
    assert dispatch1["filtered_trace"] is None

    dispatch2 = dispatches[("modeling_period_2", "trace")]
    assert dispatch2["formatter"] is None
    assert dispatch2["model"] is None
    assert dispatch2["filtered_trace"] is None

    logs = lc.items["get_energy_modeling_dispatches"]
    assert (
        "INFO - Could not determine frequency:"
        " EnergyTrace 'trace' is placeholder instance."
    ) in logs[0]
