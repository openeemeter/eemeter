import pytest

from eemeter.structures import (
    Project,
    ZIPCodeSite,
    EnergyTraceSet,
)

from eemeter.processors.collector import LogCollector
from eemeter.processors.location import get_weather_source


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

    lc = LogCollector()

    with lc.collect_logs("get_weather_source") as logger:
        ws = get_weather_source(logger, project)

    assert ws.station == '722880'

    logs = lc.items["get_weather_source"].splitlines()
    assert "INFO - Mapped ZIP code 91104 to ISD station 722880" in logs[0]
    assert "INFO - Created ISDWeatherSource using station 722880" in logs[1]


def test_bad_zip(project_bad_zip):

    lc = LogCollector()

    with lc.collect_logs("get_weather_source") as logger:
        ws = get_weather_source(logger, project_bad_zip)

    assert ws is None

    logs = lc.items["get_weather_source"].splitlines()
    assert "ERROR - Could not find ISD station for zipcode 00000" in logs[0]
