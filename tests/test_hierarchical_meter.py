from eemeter.config.yaml_parser import load

import pytest

def test_sequential_meter():
    meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {},
            ]
        }"""

    meter = load(meter_yaml)

    result = meter.evaluate(value=10)

    assert result["value"] == 10

def test_input_output_mappings():
    meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_one":"value"},
                    output_mapping: {"value":"value_one"}
                },
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_two":"value"},
                    output_mapping: {"value":"value_two"}
                },
            ]
        }"""

    meter = load(meter_yaml)

    result = meter.evaluate(value_one=10,value_two=100)

    assert result["value_one"] == 10
    assert result["value_two"] == 100

def test_incorrect_input_mappings():
    meter_yaml = """ !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_one":"value"},
                    output_mapping: {"value":"value"}
                },
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_two":"value"},
                    output_mapping: {"value":"value_one"}
                },
            ]
        }"""

    meter = load(meter_yaml)

    with pytest.raises(ValueError):
        result = meter.evaluate(value_one=10,value_two=100)

def test_incorrect_output_mappings():
    meter_yaml = """ !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_one":"value"},
                    output_mapping: {"value":"value_one"}
                },
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_two":"value"},
                    output_mapping: {"value":"value_one"}
                },
            ]
        }"""

    meter = load(meter_yaml)

    with pytest.raises(ValueError):
        result = meter.evaluate(value_one=10,value_two=100)
