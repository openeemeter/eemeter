import yaml
import tempfile
import os
from decimal import Decimal

from eemeter.config.yaml_parser import load_path,load

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

    loaded2 = load("!obj:decimal.Decimal { value : '1.23' }")
    assert isinstance(loaded2, Decimal)

    loaded3 = load("!obj:decimal.Decimal {'value' : '1.23'}")
    assert isinstance(loaded3, Decimal)

    loaded4 = load("""
                   !obj:decimal.Decimal
                     value: '1.23'
                   """)
    assert isinstance(loaded4, Decimal)
