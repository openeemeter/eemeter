from datetime import datetime
import gzip
from pkg_resources import resource_stream
import pytz

import pytest
from numpy.testing import assert_allclose

from eemeter.io.parsers import ESPIUsageParser


@pytest.fixture
def espi_electricity_xml():
    with resource_stream('eemeter.testing.resources', 'espi_electricity.xml.gz') as f:
        xml = gzip.GzipFile(fileobj=f).read()
    return xml


@pytest.fixture
def espi_natural_gas_xml():
    with resource_stream('eemeter.testing.resources', 'espi_natural_gas.xml.gz') as f:
        xml = gzip.GzipFile(fileobj=f).read()
    return xml


def test_electricity(espi_electricity_xml):
    parser = ESPIUsageParser(espi_electricity_xml)
    ets = sorted(list(parser.get_energy_trace_objects()),
                 key=lambda x: x.interpretation)

    assert len(ets) == 2

    assert ets[0].data.shape == (323, 2)
    assert ets[0].interpretation == "ELECTRICITY_CONSUMPTION_SUPPLIED"
    assert ets[0].data.index[0] == datetime(2012, 5, 2, 7, tzinfo=pytz.UTC)
    assert_allclose(ets[0].data.value.iloc[0], 0.2286)
    assert bool(ets[0].data.estimated.iloc[0]) is False

    assert ets[1].data.shape == (128, 2)
    assert ets[1].interpretation == "ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED"
    assert ets[1].data.index[0] == datetime(2015, 3, 7, 8, tzinfo=pytz.UTC)
    assert_allclose(ets[1].data.value.iloc[0], 0.0)
    assert bool(ets[1].data.estimated.iloc[0]) is False


def test_natural_gas(espi_natural_gas_xml):
    parser = ESPIUsageParser(espi_natural_gas_xml)
    ets = sorted(list(parser.get_energy_trace_objects()),
                 key=lambda x: x.interpretation)

    assert len(ets) == 1

    assert ets[0].data.shape == (3, 2)
    assert ets[0].interpretation == "NATURAL_GAS_CONSUMPTION_SUPPLIED"
    assert ets[0].data.index[0] == datetime(2012, 5, 2, 7, 0, 1, tzinfo=pytz.UTC)
    assert_allclose(ets[0].data.value.iloc[0], 0.0)
    assert bool(ets[0].data.estimated.iloc[0]) is False
