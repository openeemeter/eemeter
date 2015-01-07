from eemeter.config.yaml_parser import load_path,load

import yaml
import tempfile
import os
from decimal import Decimal

import pytest

@pytest.fixture(params=["!obj:decimal.Decimal { value : '1.23' }",
                        "!obj:decimal.Decimal {'value' : '1.23'}",
                        """!obj:decimal.Decimal
                              value: '1.23'"""])
def simple_yaml(request):
    return request.param

def test_load_path():
    fd, fname = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write("a: 23")
    loaded = load_path(fname)
    assert loaded['a'] == 23
    os.remove(fname)

def test_obj():
    loaded1 = load("a: !obj:decimal.Decimal { value : '1.23' }")
    assert isinstance(loaded1['a'], Decimal)

def test_obj_formats(simple_yaml):
    loaded = load(simple_yaml)
    assert isinstance(loaded, Decimal)
