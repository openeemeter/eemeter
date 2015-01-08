from eemeter.config.yaml_parser import load

import pytest

def test_simple_meter():
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
