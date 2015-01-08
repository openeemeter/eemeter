from eemeter.config.yaml_parser import load

import pytest

def test_simple_meter():
    meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {
                    output_mapping: {"value":"value_one"}
                },
                !obj:eemeter.meter.DummyMeter {
                    output_mapping: {"value":"value_two"}
                },
            ]
        }"""

    meter = load(meter_yaml)

    result = meter.evaluate()
