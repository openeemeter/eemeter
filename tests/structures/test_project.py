from eemeter.structures import (
    Project,
    ZIPCodeSite,
    Intervention,
    EnergyTraceSet,
)
from datetime import datetime
import pytz
import pytest


@pytest.fixture
def energy_trace_set():
    return EnergyTraceSet([])


@pytest.fixture
def interventions():
    return []

@pytest.fixture
def site():
    return ZIPCodeSite("11111")


def test_create(energy_trace_set, interventions, site):
    project = Project(energy_trace_set, interventions, site)
    assert project.energy_trace_set == energy_trace_set
    assert project.interventions == interventions
    assert project.site == site
