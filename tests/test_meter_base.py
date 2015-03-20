from eemeter.config.yaml_parser import load

from eemeter.meter import DummyMeter
from eemeter.meter import Sequence
from eemeter.meter import Condition

import pytest

def test_meter_base():
    meter_yaml = """
        !obj:eemeter.meter.DummyMeter {
            extras: {
                has_extra: true,
            },
            output_mapping: {
                has_extra: has_extra,
            },
        }
    """
    meter = load(meter_yaml)

    result = meter.evaluate(value="dummy")

    assert result["result"] == "dummy"
    assert result["has_extra"]

def test_input_output_mappings():
    meter_yaml = """
        !obj:eemeter.meter.Sequence {
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
    meter_yaml = """ !obj:eemeter.meter.Sequence {
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

    with pytest.raises(ValueError):
        result = meter.evaluate(value_one=10,value_two=100)

def test_incorrect_output_mappings():
    meter_yaml = """ !obj:eemeter.meter.Sequence {
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

def test_get_meter_input():
    dummy_meter = DummyMeter()
    assert dummy_meter.get_inputs() == {"DummyMeter":{"inputs":["value"],"child_inputs":[]}}

    seq_meter = Sequence(sequence=[DummyMeter(),DummyMeter()])
    assert seq_meter.get_inputs() == {"Sequence":
            {"inputs": [],
             "child_inputs": [
                 {"DummyMeter":{"inputs":["value"],"child_inputs":[]}},
                 {"DummyMeter":{"inputs":["value"],"child_inputs":[]}}
             ]}}

    cond_meter = Condition(condition_parameter=(lambda x: True),success=DummyMeter(),failure=DummyMeter())
    assert cond_meter.get_inputs() == {'Condition':
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
    assert "expected argument 'value' for meter 'DummyMeter'; got kwargs=[] (with mapped_inputs=[]) instead." in excinfo.value.args[0]

    seq_meter = Sequence(sequence=[
        DummyMeter(input_mapping={"value_one":"value"},
                   output_mapping={"result":"result_one"}),
        DummyMeter(input_mapping={"value_two":"value"},
                   output_mapping={"result":"result_two"}),
        ])

    assert seq_meter.evaluate(value_one=1,value_two=2) == {'result_one': 1, 'result_two': 2}

    with pytest.raises(TypeError) as excinfo:
        seq_meter.evaluate(value_one=1)
    assert "expected argument 'value' for meter 'DummyMeter';" \
           " got kwargs=[('result_one', 1), ('value_one', 1)] " \
           "(with mapped_inputs=[('result_one', 1), ('value_one', 1)]) instead." \
                   == excinfo.value.args[0]

def test_null_key_mapping():
    meter1 = Sequence(sequence=[
        DummyMeter(input_mapping={"value_one":"value"},
                   output_mapping={"result":"result_one"}),
        DummyMeter(input_mapping={"result_one":None, "value_one":"value"},
                   output_mapping={"result":"result_two"}),
        DummyMeter(input_mapping={"result_one":"value"},
                   output_mapping={"result":"result_three"}),
        ])

    result1 = meter1.evaluate(value_one=1)
    assert result1["result_one"] == 1
    assert result1["result_two"] == 1
    assert result1["result_three"] == 1

    meter2 = Sequence(sequence=[
        DummyMeter(input_mapping={"value_one": "value"},
                   output_mapping={"result": None}),
        DummyMeter(input_mapping={"result": "value"},
                   output_mapping={"result": "result_one"}),
        ])


    with pytest.raises(TypeError):
        result2 = meter2.evaluate(value_one=1)

def test_multiple_mapping():
    meter = Sequence(sequence=[
        DummyMeter(input_mapping={"value_one": "value"},
                   output_mapping={"result": ["result_one","result_two"]}),
        DummyMeter(input_mapping={"result_one": "value"},
                   output_mapping={"result": "result_three"}),
        DummyMeter(input_mapping={"result_two": "value"},
                   output_mapping={"result": "result_four"}),
        ])

    result = meter.evaluate(value_one=1)
    assert result["result_one"] == 1
    assert result["result_two"] == 1
    assert result["result_three"] == 1
    assert result["result_four"] == 1


