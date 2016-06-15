from eemeter.structures import EnergyTraceSet
from eemeter.structures import EnergyTrace
import pandas as pd
import numpy as np

import pytest

@pytest.fixture
def trace1():

    columns = {"value": [1, np.nan], "estimated": [False, False]}
    column_names = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=2, freq='D')
    data = pd.DataFrame(columns, index=index, columns=column_names)

    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", data=data,
                       unit="KWH")

@pytest.fixture
def trace2():

    columns = {"value": [1, np.nan], "estimated": [False, False]}
    column_names = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=2, freq='D')
    data = pd.DataFrame(columns, index=index, columns=column_names)

    return EnergyTrace("ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED", data=data,
                       unit="KWH")

def test_create_dict(trace1, trace2):
    trace_dict = {"t1": trace1, "t2": trace2}
    ets = EnergyTraceSet(trace_dict)
    assert len(list(ets.itertraces())) == 2
    assert list(ets.itertraces())[0][0] in ['t1', 't2']
    assert list(ets.itertraces())[1][0] in ['t1', 't2']

def test_create_dict_with_labels(trace1, trace2):
    trace_dict = {"t1": trace1, "t2": trace2}

    with pytest.warns(UserWarning):
        ets = EnergyTraceSet(trace_dict, labels=["t1", "t2"])

def test_create_list(trace1, trace2):
    traces = [trace1, trace2]
    labels = ["t1", "t2"]

    ets = EnergyTraceSet(traces, labels)

    assert len(list(ets.itertraces())) == 2
    assert list(ets.itertraces())[0][0] in ['t1', 't2']
    assert list(ets.itertraces())[1][0] in ['t1', 't2']

def test_create_list_mismatch_labels(trace1, trace2):
    traces = [trace1, trace2]
    labels = ["t1", "t2", "t3"]

    with pytest.raises(ValueError):
        ets = EnergyTraceSet(traces, labels)

def test_create_list_double_labels(trace1, trace2):
    traces = [trace1, trace2]
    labels = ["t1", "t1"]

    with pytest.raises(ValueError):
        ets = EnergyTraceSet(traces, labels)

def test_create_list_no_labels(trace1, trace2):
    traces = [trace1, trace2]
    ets = EnergyTraceSet(traces)

    assert len(list(ets.itertraces())) == 2
    assert list(ets.itertraces())[0][0] in ['0', '1']
    assert list(ets.itertraces())[1][0] in ['0', '1']
