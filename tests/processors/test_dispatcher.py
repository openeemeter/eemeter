from datetime import datetime
import pytz

import pytest
import numpy as np
import pandas as pd

from eemeter.processors.dispatchers import dispatch_energy_modelers
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


def test_basic_usage(modeling_period_set, trace_set):
    lc = LogCollector()

    with lc.collect_logs("dispatch_energy_modelers") as logger:
        dispatches = list(dispatch_energy_modelers(
            logger, modeling_period_set, trace_set))

    assert len(dispatches) == 2
    assert isinstance(dispatches[0][0][0], ModelDataFormatter)
    assert isinstance(dispatches[0][0][1], SeasonalElasticNetCVModel)
    assert isinstance(dispatches[0][0][2], ModelingPeriod)
    assert isinstance(dispatches[0][0][3], EnergyTrace)
    assert dispatches[0][1] == "modeling_period_1"
    assert dispatches[0][2] == "trace"

    assert isinstance(dispatches[1][0][0], ModelDataFormatter)
    assert isinstance(dispatches[1][0][1], SeasonalElasticNetCVModel)
    assert isinstance(dispatches[1][0][2], ModelingPeriod)
    assert isinstance(dispatches[1][0][3], EnergyTrace)
    assert dispatches[1][1] == "modeling_period_2"
    assert dispatches[1][2] == "trace"
