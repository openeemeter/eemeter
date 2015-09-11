from eemeter.config.yaml_parser import load

from eemeter.meter import MeterBase
from eemeter.meter import DataContainer
from eemeter.meter import DataCollection

from eemeter.meter import DummyMeter
from eemeter.meter import Sequence
from eemeter.meter import Condition

import pytest

@pytest.fixture
def data_container_name_value():
    value = DataContainer(name="name", value="value", tags=["tag"])
    return value

@pytest.fixture
def data_collection(data_container_name_value):
    dc = DataCollection()
    dc.add_data(data_container_name_value)
    return dc

def test_data_container(data_container_name_value):
    assert data_container_name_value.name == "name"
    assert data_container_name_value.value == "value"
    assert data_container_name_value.get_value() == "value"

    data_container_name_value.set_value("value1")
    assert data_container_name_value.get_value() == "value1"

    assert type(data_container_name_value.tags) == frozenset
    assert "tag" in data_container_name_value.tags

def test_data_container_init_tag_type_errors():
    with pytest.raises(TypeError):
        DataContainer(name="name", value="value", tags="tag")

def test_data_collection_init_tag_type_errors():
    with pytest.raises(TypeError):
        DataCollection(tags="tag")

def test_data_container_add_tags(data_container_name_value):
    assert data_container_name_value.tags == frozenset(["tag"])

    data_container_name_value.add_tags(["tag1"])

    assert data_container_name_value.tags == frozenset(["tag","tag1"])

    with pytest.raises(TypeError):
        data_container_name_value.add_tags("tag2")

def test_data_collection_add_tags(data_collection):
    assert data_collection.get_data("name").tags == frozenset(["tag"])
    data_collection.add_tags(["tag1"])
    assert data_collection.get_data("name").tags == frozenset(["tag","tag1"])

    with pytest.raises(TypeError):
        data_collection.add_tags("tag2")

def test_data_collection_count(data_collection):
    assert data_collection.count() == 1

def test_data_collection_get_data(data_collection):
    assert data_collection.get_data("name").name == "name"
    assert data_collection.get_data("name", tags=["tag"]).name == "name"
    assert data_collection.get_data("name", tags=["nonexistent"]) is None
    assert data_collection.get_data("nonexistent") is None

    with pytest.raises(TypeError):
        data_collection.get_data("name", "tag2")

def test_data_collection_add_data(data_collection):
    # before
    assert data_collection.get_data("new_data") == None

    # add new data
    new_data = DataContainer("new_data", value="new_data_value", tags=["new_tag"])
    data_collection.add_data(new_data)

    # after
    assert data_collection.get_data("new_data").name == "new_data"
    assert data_collection.get_data("new_data").value == "new_data_value"
    assert data_collection.get_data("new_data", tags=["new_tag"]).name == "new_data"
    assert data_collection.get_data("new_data", tags=["bad_tag"]) == None

    with pytest.raises(TypeError):
        data_collection.get_data("new_data", "tag2")

def test_data_collection_add_overlapping_data(data_collection):
    # before
    assert data_collection.get_data("name") is not None

    # attempt to add overlapping
    new_data = DataContainer("name", "new_data_value")
    with pytest.raises(ValueError) as e:
        data_collection.add_data(new_data)
        assert "Element already exists in data container:" in e.message

    # not overlapping (different tag) so this is ok.
    new_data = DataContainer("name", "new_data_value", ["tag1"])
    data_collection.add_data(new_data)

    assert data_collection.get_data("name", tags=["tag1"]).name == "name"

def test_data_collection_add_data_collection(data_collection):
    dc = DataCollection()
    new_data = DataContainer("name", "new_data_value", ["tag1"])
    dc.add_data(new_data)
    data_collection.add_data_collection(dc,tagspace=["tagspace"])

    assert data_collection.get_data("name", tags=["tag1","tagspace"]).name == "name"

    with pytest.raises(TypeError):
        data_collection.add_data_collection("name", "tagspace")


def test_data_collection_search(data_collection):
    assert data_collection.search("name").count() == 1
    assert data_collection.search("nom").count() == 0
    assert data_collection.search("nam").count() == 0
    assert data_collection.search("name", tags=[]).count() == 1
    assert data_collection.search("name", tags=["tag"]).count() == 1
    assert data_collection.search("name", tags=["dag"]).count() == 0
    assert data_collection.search("name", tags=["tag", "dag"]).count() == 1

def test_data_collection_filter_by_tag(data_collection):
    assert data_collection.filter_by_tag(["tag"]).count() == 1
    assert data_collection.filter_by_tag(["tag","tag1"]).count() == 0
    assert data_collection.filter_by_tag(["tag1"]).count() == 0
    assert data_collection.filter_by_tag([]).count() == 1
    assert data_collection.filter_by_tag(None).count() == 1

def test_insufficient_query(data_collection):
    new_data = DataContainer("name", "new_data_value", ["tag1"])
    data_collection.add_data(new_data)

    # should be two elements with name "name"
    with pytest.raises(ValueError) as e:
        data_collection.get_data("name")
        assert "Ambiguous criteria:" in e.message

def test_data_collection_iteritems(data_collection):
    for item in data_collection.iteritems():
        assert item.name == "name"

def test_data_collection_copy(data_collection):
    for item1,item2 in zip(data_collection.iteritems(),
            data_collection.copy().iteritems()):
        assert item1.name == item2.name

def test_data_collection_creation_shortcut():
    dc = DataCollection(item1="item1_value", item2="item2_value", tags=["tag"])
    assert dc.get_data("item1",tags=["tag"]).name == "item1"
    assert dc.get_data("item1",tags=["tag"]).value == "item1_value"
    assert dc.get_data("item2",tags=["tag"]).name == "item2"
    assert dc.get_data("item2",tags=["tag"]).value == "item2_value"

def test_dummy_meter(data_collection):
    meter_yaml = """
        !obj:eemeter.meter.DummyMeter {
            input_mapping: { "value": { "name":"name", }, },
            output_mapping: { "result": { "name":"result", }, },
        }
    """

    meter = load(meter_yaml)
    result = meter.evaluate(data_collection)

    assert result.get_data(name="result").value == "value"
    assert result.get_data(name="name") == None

def test_dummy_meter_output_duplication_list(data_collection):
    meter_yaml = """
        !obj:eemeter.meter.DummyMeter {
            input_mapping: { "value": { "name":"name", }, },
            output_mapping: {
                "result": [
                    { "name":"result1", },
                    { "name":"result2", },
                ]
            },
        }
    """

    meter = load(meter_yaml)
    result = meter.evaluate(data_collection)

    assert result.get_data(name="result1").value == "value"
    assert result.get_data(name="result2").value == "value"
    assert result.get_data(name="name") == None


def test_dummy_meter_tags(data_collection):
    meter_yaml = """
        !obj:eemeter.meter.DummyMeter {
            input_mapping: {
                "value": {
                    "name":"name",
                    "tags": ["tag"],
                },
            },
            output_mapping: {
                "result": {
                    "name":"result_1",
                    "tags": ["tag_1"],
                },
            },
        }
    """

    meter = load(meter_yaml)

    result = meter.evaluate(data_collection)

    assert result.get_data("result_1").value == "value"
    assert result.get_data("result_1",tags=["tag_1"]).value == "value"
    assert result.get_data("result_1",tags=["tag"]) == None
    assert result.get_data("result") == None
    assert result.get_data(name="name") == None

def test_dummy_meter_auxiliary_inputs():
    meter_yaml = """
        !obj:eemeter.meter.DummyMeter {
            auxiliary_inputs: { "value": "aux" },
            output_mapping: { "result": {}, },
        }
    """

    meter = load(meter_yaml)
    data_collection = DataCollection()
    result = meter.evaluate(data_collection)

    assert result.get_data(name="result").value == "aux"

def test_dummy_meter_auxiliary_outputs(data_collection):
    meter_yaml = """
        !obj:eemeter.meter.DummyMeter {
            input_mapping: { "value": { "name":"name", }, },
            output_mapping: {
                "result": { "name":"result", },
                "aux_result": { "name":"aux_result", },
            },
            auxiliary_outputs: { "aux_result": "aux" },
        }
    """

    meter = load(meter_yaml)
    result = meter.evaluate(data_collection)

    assert result.get_data(name="result").value == "value"
    assert result.get_data(name="aux_result").value == "aux"
