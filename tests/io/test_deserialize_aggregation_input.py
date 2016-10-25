import pytest

from eemeter.io.serializers import deserialize_aggregation_input


@pytest.fixture
def aggregation_input():
    aggregation_input = {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": 1,
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 12.7673693526363,
                    "baseline_lower": 6.64513428655014,
                    "baseline_upper": 16.6902599327546,
                    "baseline_n": 365.0,
                    "reporting_value": 10.4456308079608,
                    "reporting_lower": 6.87143931976653,
                    "reporting_upper": 19.4891794773991,
                    "reporting_n": 365.0
                },
                {
                    "label": 3,
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 12.7673693526363,
                    "baseline_lower": 6.64513428655014,
                    "baseline_upper": 16.6902599327546,
                    "baseline_n": 365.0,
                    "reporting_value": 10.4456308079608,
                    "reporting_lower": 6.87143931976653,
                    "reporting_upper": 19.4891794773991,
                    "reporting_n": 365.0
                }
            ]
        }
    }
    return aggregation_input


def test_basic_usage(aggregation_input):
    result = deserialize_aggregation_input(aggregation_input)

    assert result['aggregation_interpretation'] is not None
    assert result['derivative_interpretation'] is not None
    assert result['trace_interpretation'] is not None

    assert result['baseline_default_value'] is None
    assert result['reporting_default_value'] is None

    assert result['derivative_pairs'] is not None


def test_no_type(aggregation_input):
    del aggregation_input['type']

    result = deserialize_aggregation_input(aggregation_input)

    assert result["error"].startswith('Serialization "type"')


def test_unrecognized_type(aggregation_input):
    aggregation_input['type'] = "UNKNOWN"

    result = deserialize_aggregation_input(aggregation_input)

    assert result["error"].startswith('Serialization type "UNKNOWN"')


def test_no_aggregation_interpretation(aggregation_input):
    del aggregation_input['aggregation_interpretation']

    result = deserialize_aggregation_input(aggregation_input)

    assert result["error"].startswith('For serialization')


def test_no_trace_interpretation(aggregation_input):
    del aggregation_input['trace_interpretation']

    result = deserialize_aggregation_input(aggregation_input)

    assert result["error"].startswith('For serialization')


def test_no_derivative_interpretation(aggregation_input):
    del aggregation_input['derivative_interpretation']

    result = deserialize_aggregation_input(aggregation_input)

    assert result["error"].startswith('For serialization')


def test_no_derivatives(aggregation_input):
    del aggregation_input['derivatives']

    result = deserialize_aggregation_input(aggregation_input)

    assert result["error"].startswith('For serialization')


def test_bad_baseline_default(aggregation_input):
    aggregation_input['baseline_default_value'] = {}
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Serialization')

    aggregation_input['baseline_default_value'] = {"type": "Unknown"}
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Serialization')

    aggregation_input['baseline_default_value'] = {
        "type": "SIMPLE_DEFAULT",
        "value": None,
        "lower": None,
        "upper": None,
    }
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Missing')

    aggregation_input['baseline_default_value'] = {
        "type": "SIMPLE_DEFAULT",
        "value": None,
        "lower": None,
        "upper": None,
        "n": None,
    }
    result = deserialize_aggregation_input(aggregation_input)
    assert result["baseline_default_value"] is not None


def test_bad_reporting_default(aggregation_input):

    aggregation_input['reporting_default_value'] = {}
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Serialization')

    aggregation_input['reporting_default_value'] = {"type": "Unknown"}
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Serialization')

    aggregation_input['reporting_default_value'] = {
        "type": "SIMPLE_DEFAULT",
        "value": None,
        "lower": None,
        "upper": None,
    }
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Missing')

    aggregation_input['reporting_default_value'] = {
        "type": "SIMPLE_DEFAULT",
        "value": None,
        "lower": None,
        "upper": None,
        "n": None,
    }
    result = deserialize_aggregation_input(aggregation_input)
    assert result["reporting_default_value"] is not None


def test_bad_derivatives(aggregation_input):

    aggregation_input['derivatives'] = {}
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Serialization')

    aggregation_input['derivatives'] = {"type": "Unknown"}
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('Serialization')

    aggregation_input['derivatives'] = {
        "type": "DERIVATIVE_PAIRS",
    }
    result = deserialize_aggregation_input(aggregation_input)
    assert result["error"].startswith('For serialization')
