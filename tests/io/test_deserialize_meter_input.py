import pytest
import pandas as pd
import pytz

from eemeter.structures import (
    EnergyTrace,
    ModelingPeriodSet,
)

from eemeter.io.serializers import deserialize_meter_input


@pytest.fixture
def meter_input():
    meter_input = {
        "type": "SINGLE_TRACE_SIMPLE_PROJECT",
        "trace": {
            "type": "ARBITRARY_START",
            "interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
            "unit": "therm",
            "records": [
                {
                    "start": dt.isoformat(),
                    "value": 1.0,
                    "estimated": False
                }

                for dt in pd.date_range(start="2010-01-01", end="2012-01-01",
                                        freq="MS", tz=pytz.UTC)
            ]
        },
        "project": {
            "type": "PROJECT_WITH_SINGLE_MODELING_PERIOD_GROUP",
            "zipcode": "91104",
            "modeling_period_group": {
                "baseline_period": {
                    "start": None,
                    "end": "2012-01-01T00:00:00+00:00"
                },
                "reporting_period": {
                    "start": "2012-02-01T00:00:00+00:00",
                    "end": None
                }
            }
        }
    }
    return meter_input


def test_basic_usage(meter_input):
    result = deserialize_meter_input(meter_input)
    assert isinstance(result["trace"], EnergyTrace)
    assert isinstance(result["project"]["modeling_period_set"],
                      ModelingPeriodSet)
    assert isinstance(result["project"]["zipcode"],
                      str)


def test_missing_type(meter_input):
    del meter_input["type"]
    result = deserialize_meter_input(meter_input)
    assert "type" in result["error"]


def test_missing_trace(meter_input):
    del meter_input["trace"]
    result = deserialize_meter_input(meter_input)
    assert "trace" in result["error"]


def test_missing_trace_type(meter_input):
    del meter_input["trace"]["type"]
    result = deserialize_meter_input(meter_input)
    assert "type" in result["error"]


def test_missing_trace_interpretation(meter_input):
    del meter_input["trace"]["interpretation"]
    result = deserialize_meter_input(meter_input)
    assert "interpretation" in result["error"]


def test_missing_trace_unit(meter_input):
    del meter_input["trace"]["unit"]
    result = deserialize_meter_input(meter_input)
    assert "unit" in result["error"]


def test_missing_trace_records(meter_input):
    del meter_input["trace"]["records"]
    result = deserialize_meter_input(meter_input)
    assert "records" in result["error"]


def test_missing_project(meter_input):
    del meter_input["project"]
    result = deserialize_meter_input(meter_input)
    assert "project" in result["error"]


def test_missing_project_type(meter_input):
    del meter_input["project"]
    result = deserialize_meter_input(meter_input)
    assert "type" in result["error"]


def test_missing_project_zipcode(meter_input):
    del meter_input["project"]["zipcode"]
    result = deserialize_meter_input(meter_input)
    assert "zipcode" in result["error"]


def test_missing_project_modeling_period_group(meter_input):
    del meter_input["project"]["modeling_period_group"]
    result = deserialize_meter_input(meter_input)
    assert "modeling_period_group" in result["error"]
