import pytest
from eemeter.ee.derivatives import DerivativePair, Derivative
from eemeter.ee.aggregate import Aggregator


@pytest.fixture
def aggregation_input():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 3,
                    "baseline_upper": 3,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                },
                {
                    "label": "2",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 4,
                    "baseline_upper": 4,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 8,
                    "reporting_upper": 8,
                    "reporting_n": 5,
                }
            ]
        }
    }


@pytest.fixture
def aggregation_input_empty():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": []
        }
    }


@pytest.fixture
def aggregation_input_single():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 3,
                    "baseline_upper": 3,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                }
            ]
        }
    }


@pytest.fixture
def aggregation_input_mixed_derivative_interpretation():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 3,
                    "baseline_upper": 3,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                },
                {
                    "label": "2",
                    "derivative_interpretation": "gross_predicted",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 4,
                    "baseline_upper": 4,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 8,
                    "reporting_upper": 8,
                    "reporting_n": 5,
                }
            ]
        }
    }


@pytest.fixture
def aggregation_input_mixed_trace_interpretation():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 3,
                    "baseline_upper": 3,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                },
                {
                    "label": "2",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "ELECTRICITY_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 4,
                    "baseline_upper": 4,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 8,
                    "reporting_upper": 8,
                    "reporting_n": 5,
                }
            ]
        }
    }


@pytest.fixture
def aggregation_input_mixed_unit():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 3,
                    "baseline_upper": 3,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                },
                {
                    "label": "2",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "kWh",
                    "baseline_value": 10,
                    "baseline_lower": 4,
                    "baseline_upper": 4,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 8,
                    "reporting_upper": 8,
                    "reporting_n": 5,
                }
            ]
        }
    }


@pytest.fixture
def aggregation_input_one_invalid():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": None,
                    "baseline_lower": None,
                    "baseline_upper": None,
                    "baseline_n": None,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                },
                {
                    "label": "2",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 4,
                    "baseline_upper": 4,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 8,
                    "reporting_upper": 8,
                    "reporting_n": 5,
                }
            ]
        }
    }


@pytest.fixture
def aggregation_input_one_invalid_default_baseline():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "baseline_default_value": {
            "type": "SIMPLE_DEFAULT",
            "value": 0,
            "lower": 0,
            "upper": 0,
            "n": 0,
        },
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": None,
                    "baseline_lower": None,
                    "baseline_upper": None,
                    "baseline_n": None,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                },
                {
                    "label": "2",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 4,
                    "baseline_upper": 4,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 8,
                    "reporting_upper": 8,
                    "reporting_n": 5,
                }
            ]
        }
    }


@pytest.fixture
def aggregation_input_one_invalid_default_reporting():
    return {
        "type": "BASIC_AGGREGATION",
        "aggregation_interpretation": "SUM",
        "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
        "derivative_interpretation": "annualized_weather_normal",
        "reporting_default_value": {
            "type": "SIMPLE_DEFAULT",
            "value": 0,
            "lower": 0,
            "upper": 0,
            "n": 0,
        },
        "derivatives": {
            "type": "DERIVATIVE_PAIRS",
            "derivative_pairs": [
                {
                    "label": "1",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 3,
                    "baseline_upper": 3,
                    "baseline_n": 5,
                    "reporting_value": 10,
                    "reporting_lower": 6,
                    "reporting_upper": 6,
                    "reporting_n": 5,
                },
                {
                    "label": "2",
                    "derivative_interpretation": "annualized_weather_normal",
                    "trace_interpretation": "NATURAL_GAS_CONSUMPTION_SUPPLIED",
                    "unit": "therm",
                    "baseline_value": 10,
                    "baseline_lower": 4,
                    "baseline_upper": 4,
                    "baseline_n": 5,
                    "reporting_value": None,
                    "reporting_lower": None,
                    "reporting_upper": None,
                    "reporting_n": None,
                }
            ]
        }
    }


def test_basic_usage(aggregation_input):

    aggregator = Aggregator()

    output = aggregator.aggregate(aggregation_input)

    status = output['status']
    assert status['1']['status'] == "ACCEPTED"
    assert status['1']['baseline_status'] == "ACCEPTED"
    assert status['1']['reporting_status'] == "ACCEPTED"
    assert status['2']['status'] == "ACCEPTED"
    assert status['2']['baseline_status'] == "ACCEPTED"
    assert status['2']['reporting_status'] == "ACCEPTED"

    derivative_pair = output["aggregated"]

    assert derivative_pair["label"] is None

    assert derivative_pair["derivative_interpretation"] == \
        "annualized_weather_normal"

    assert derivative_pair["trace_interpretation"] == \
        "NATURAL_GAS_CONSUMPTION_SUPPLIED"

    baseline = derivative_pair["baseline"]
    assert baseline["value"] == 20
    assert baseline["lower"] == 5
    assert baseline["upper"] == 5
    assert baseline["n"] == 10

    reporting = derivative_pair["reporting"]
    assert reporting["value"] == 20
    assert reporting["lower"] == 10
    assert reporting["upper"] == 10
    assert reporting["n"] == 10


def test_empty(aggregation_input_empty):

    aggregator = Aggregator()
    with pytest.raises(ValueError):
        aggregator.aggregate(aggregation_input_empty)


def test_single(aggregation_input_single):

    aggregator = Aggregator()
    output = aggregator.aggregate(aggregation_input_single)

    status = output['status']
    assert status['1']['baseline_status'] == "ACCEPTED"
    assert status['1']['reporting_status'] == "ACCEPTED"


def test_mixed_derivative_interpretaiton_fails(
        aggregation_input_mixed_derivative_interpretation):

    aggregator = Aggregator()
    with pytest.raises(ValueError):
        aggregator.aggregate(aggregation_input_mixed_derivative_interpretation)


def test_mixed_trace_interpretaiton_fails(
        aggregation_input_mixed_trace_interpretation):

    aggregator = Aggregator()
    with pytest.raises(ValueError):
        aggregator.aggregate(aggregation_input_mixed_trace_interpretation)


def test_mixed_unit_fails(aggregation_input_mixed_unit):

    aggregator = Aggregator()
    with pytest.raises(ValueError):
        aggregator.aggregate(aggregation_input_mixed_unit)


def test_missing(aggregation_input_one_invalid):

    aggregator = Aggregator()
    output = aggregator.aggregate(aggregation_input_one_invalid)

    status = output['status']
    assert status['1']['status'] == "REJECTED"
    assert status['1']['baseline_status'] == "REJECTED"
    assert status['1']['reporting_status'] == "ACCEPTED"
    assert status['2']['status'] == "ACCEPTED"
    assert status['2']['baseline_status'] == "ACCEPTED"
    assert status['2']['reporting_status'] == "ACCEPTED"


def test_missing_with_baseline_default(
        aggregation_input_one_invalid_default_baseline):

    aggregator = Aggregator()
    output = aggregator.aggregate(
        aggregation_input_one_invalid_default_baseline)

    status = output['status']
    assert status['1']['baseline_status'] == "DEFAULT"
    assert status['1']['reporting_status'] == "ACCEPTED"
    assert status['2']['baseline_status'] == "ACCEPTED"
    assert status['2']['reporting_status'] == "ACCEPTED"


def test_missing_with_reporting_default(
        aggregation_input_one_invalid_default_reporting):

    aggregator = Aggregator()
    output = aggregator.aggregate(
        aggregation_input_one_invalid_default_reporting)

    status = output['status']
    assert status['1']['baseline_status'] == "ACCEPTED"
    assert status['1']['reporting_status'] == "ACCEPTED"
    assert status['2']['baseline_status'] == "ACCEPTED"
    assert status['2']['reporting_status'] == "DEFAULT"

