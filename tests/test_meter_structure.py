from eemeter.config.yaml_parser import load
from eemeter.meter import DummyMeter
from eemeter.meter import SequentialMeter
from eemeter.meter import ConditionalMeter

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

    assert result["result"] == 10

def test_input_output_mappings():
    meter_yaml = """
        !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_one":"value"},
                    output_mapping: {"result":"result_one"}
                },
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_two":"value"},
                    output_mapping: {"result":"result_two"}
                },
            ]
        }"""

    meter = load(meter_yaml)

    result = meter.evaluate(value_one=10,value_two=100)

    assert result["result_one"] == 10
    assert result["result_two"] == 100

def test_incorrect_input_mappings():
    meter_yaml = """ !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_one":"value"},
                    output_mapping: {"result":"value"}
                },
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_two":"value"},
                    output_mapping: {"result":"value_one"}
                },
            ]
        }"""

    meter = load(meter_yaml)

    print("start")
    with pytest.raises(ValueError):
        result = meter.evaluate(value_one=10,value_two=100)
    print("end")

def test_incorrect_output_mappings():
    meter_yaml = """ !obj:eemeter.meter.SequentialMeter {
            sequence: [
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_one":"value"},
                    output_mapping: {"result":"value_one"}
                },
                !obj:eemeter.meter.DummyMeter {
                    input_mapping: {"value_two":"value"},
                    output_mapping: {"result":"value_one"}
                },
            ]
        }"""

    meter = load(meter_yaml)

    with pytest.raises(ValueError):
        result = meter.evaluate(value_one=10,value_two=100)

def test_conditional_meter():
    meter_yaml="""
        !obj:eemeter.meter.ConditionalMeter {
            condition_parameter: "electricity_present",
            success: !obj:eemeter.meter.DummyMeter {
                input_mapping: {"success":"value"},
            },
            failure: !obj:eemeter.meter.DummyMeter {
                input_mapping: {"failure":"value"},
            },
        }
        """
    meter = load(meter_yaml)
    assert meter.evaluate(electricity_present=True,success="success",failure="failure")["result"] == "success"
    assert meter.evaluate(electricity_present=False,success="success",failure="failure")["result"] == "failure"

def test_conditional_meter_without_params():
    meter_yaml="""
        !obj:eemeter.meter.ConditionalMeter {
            condition_parameter: "electricity_present",
        }
        """
    meter = load(meter_yaml)
    assert isinstance(meter.evaluate(electricity_present=True),dict)
    assert isinstance(meter.evaluate(electricity_present=False),dict)

def test_debug_meter():

    meter_yaml="""
        !obj:eemeter.meter.DebugMeter {
        }
        """
    meter = load(meter_yaml)

def test_get_meter_input():
    dummy_meter = DummyMeter()
    assert dummy_meter.get_inputs() == {"DummyMeter":{"inputs":["value"],"child_inputs":[]}}

    seq_meter = SequentialMeter(sequence=[DummyMeter(),DummyMeter()])
    assert seq_meter.get_inputs() == {"SequentialMeter":
            {"inputs": [],
             "child_inputs": [
                 {"DummyMeter":{"inputs":["value"],"child_inputs":[]}},
                 {"DummyMeter":{"inputs":["value"],"child_inputs":[]}}
             ]}}

    cond_meter = ConditionalMeter(condition_parameter=(lambda x: True),success=DummyMeter(),failure=DummyMeter())
    assert cond_meter.get_inputs() == {'ConditionalMeter':
            {'child_inputs':
                {'failure':
                    {'DummyMeter': {'child_inputs': [], 'inputs': ['value']}},
                 'success':
                    {'DummyMeter': {'child_inputs': [],'inputs': ['value']}}},
             'inputs': []
             }}

def test_sane_missing_input_error_messages():
    dummy_meter = DummyMeter()
    with pytest.raises(TypeError) as excinfo:
        dummy_meter.evaluate()
    assert "expected argument 'value' for meter 'DummyMeter'; got kwargs={} (with mapped_inputs={}) instead." in excinfo.value.args[0]

    seq_meter = SequentialMeter(sequence=[
        DummyMeter(input_mapping={"value_one":"value"},
                   output_mapping={"result":"result_one"}),
        DummyMeter(input_mapping={"value_two":"value"},
                   output_mapping={"result":"result_two"}),
        ])

    assert seq_meter.evaluate(value_one=1,value_two=2) == {'result_one': 1, 'result_two': 2}

    with pytest.raises(TypeError) as excinfo:
        seq_meter.evaluate(value_one=1)
    assert "expected argument 'value' for meter 'DummyMeter';" \
           " got kwargs={'result_one': 1, 'value_one': 1} " \
           "(with mapped_inputs={'result_one': 1, 'value_one': 1}) instead." \
                   == excinfo.value.args[0]

