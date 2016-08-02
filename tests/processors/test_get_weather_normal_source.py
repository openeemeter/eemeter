import pytest

from eemeter.structures import (
    Project,
    ZIPCodeSite,
    EnergyTraceSet,
)

from eemeter.processors.location import get_weather_normal_source


@pytest.fixture
def project():
    ets = EnergyTraceSet({})
    interventions = []
    site = ZIPCodeSite("91104")
    project = Project(ets, interventions, site)
    return project


@pytest.fixture
def project_bad_zip():
    ets = EnergyTraceSet({})
    interventions = []
    site = ZIPCodeSite("00000")
    project = Project(ets, interventions, site)
    return project


def test_basic_usage(project):

    ws = get_weather_normal_source(project)

    assert ws.station == '722880'


def test_bad_zip(project_bad_zip):

    ws = get_weather_normal_source(project_bad_zip)

    assert ws is None
