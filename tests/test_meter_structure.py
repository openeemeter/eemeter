from eemeter.config.yaml_parser import load
from eemeter.meter import DummyMeter
from eemeter.meter import Sequence
from eemeter.meter import Condition
from eemeter.meter import And
from eemeter.meter import Or
from eemeter.meter import ForEachFuelType

import pytest

def test_sequential_meter():
    meter_yaml = """
        !obj:eemeter.meter.Sequence {
            sequence: [
                !obj:eemeter.meter.DummyMeter {},
            ]
        }"""

    meter = load(meter_yaml)

    result = meter.evaluate(value=10)

    assert result["result"] == 10

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

def test_conditional_meter():
    meter_yaml="""
        !obj:eemeter.meter.Condition {
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
        !obj:eemeter.meter.Condition {
            condition_parameter: "electricity_present",
        }
        """
    meter = load(meter_yaml)
    assert isinstance(meter.evaluate(electricity_present=True),dict)
    assert isinstance(meter.evaluate(electricity_present=False),dict)

def test_debug_meter():

    meter_yaml="""
        !obj:eemeter.meter.Debug {
        }
        """
    meter = load(meter_yaml)

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

def test_and_meter():
    with pytest.raises(ValueError):
        meter0 = And(inputs=[])

    meter1 = And(inputs=["result_one"])
    assert meter1.evaluate(result_one=True,result_two=True)["output"]

    meter2 = And(inputs=["result_one","result_two"])
    assert meter2.evaluate(result_one=True,result_two=True)["output"]
    assert not meter2.evaluate(result_one=False,result_two=True)["output"]
    assert not meter2.evaluate(result_one=True,result_two=False)["output"]
    assert not meter2.evaluate(result_one=False,result_two=False)["output"]

    meter3 = And(inputs=["result_one","result_two","result_three"])
    with pytest.raises(ValueError):
        assert meter3.evaluate(result_one=True,result_two=True)
    assert meter3.evaluate(result_one=True,result_two=True,result_three=True)["output"]
    assert not meter3.evaluate(result_one=True,result_two=True,result_three=False)["output"]
    assert not meter3.evaluate(result_one=False,result_two=True,result_three=True)["output"]


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

def test_or_meter():
    with pytest.raises(ValueError):
        meter0 = Or(inputs=[])

    meter1 = Or(inputs=["result_one"])
    assert meter1.evaluate(result_one=True,result_two=True)["output"]

    meter2 = Or(inputs=["result_one","result_two"])
    assert meter2.evaluate(result_one=True,result_two=True)["output"]
    assert meter2.evaluate(result_one=False,result_two=True)["output"]
    assert meter2.evaluate(result_one=True,result_two=False)["output"]
    assert not meter2.evaluate(result_one=False,result_two=False)["output"]

    meter3 = Or(inputs=["result_one","result_two","result_three"])
    with pytest.raises(ValueError):
        assert meter3.evaluate(result_one=True,result_two=True)
    assert meter3.evaluate(result_one=True,result_two=True,result_three=True)["output"]
    assert meter3.evaluate(result_one=True,result_two=True,result_three=False)["output"]
    assert meter3.evaluate(result_one=False,result_two=True,result_three=True)["output"]
    assert not meter3.evaluate(result_one=False,result_two=False,result_three=False)["output"]

def test_for_each_fuel_type():
    meter_yaml = """
        !obj:eemeter.meter.ForEachFuelType {
            fuel_types: [electricity,natural_gas],
            meter: !obj:eemeter.meter.Sequence {
                sequence: [
                    !obj:eemeter.meter.DummyMeter {
                        input_mapping: {
                            fuel_type: value,
                        }
                    },
                    !obj:eemeter.meter.DummyMeter {
                        input_mapping: {
                            value_one: value,
                        },
                        output_mapping: {
                            result: result_one
                        }
                    }
                ]
            }
        }
    """
    meter = load(meter_yaml)

    result = meter.evaluate(value_one=1)

    assert result["result_electricity"] == "electricity"
    assert result["result_natural_gas"] == "natural_gas"
    assert result["result_one_electricity"] == 1
    assert result["result_one_natural_gas"] == 1

def test_switch():
    meter_yaml = """
        !obj:eemeter.meter.Switch {
            target: target,
            cases: {
                1: !obj:eemeter.meter.DummyMeter {
                    input_mapping: {
                        value_one: value,
                    },
                },
                2: !obj:eemeter.meter.DummyMeter {
                    input_mapping: {
                        value_two: value,
                    },
                },
                3: !obj:eemeter.meter.DummyMeter {
                    input_mapping: {
                        value_three: value,
                    },
                },
            },
            default: !obj:eemeter.meter.DummyMeter {
                input_mapping: {
                    value_default: value,
                },
            }
        }
    """

    meter = load(meter_yaml)

    result1 = meter.evaluate(target=1,value_one=1,value_two=2,value_three=3,value_default=4)
    result2 = meter.evaluate(target=2,value_one=1,value_two=2,value_three=3,value_default=4)
    result3 = meter.evaluate(target=3,value_one=1,value_two=2,value_three=3,value_default=4)
    result4 = meter.evaluate(target=4,value_one=1,value_two=2,value_three=3,value_default=4)
    result5 = meter.evaluate(value_one=1,value_two=2,value_three=3,value_default=4)

    assert 1 == result1.get("result")
    assert 2 == result2.get("result")
    assert 3 == result3.get("result")
    assert 4 == result4.get("result")
    assert None == result5.get("result")

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
