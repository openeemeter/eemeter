from eemeter.structures import (
    Project,
    ZIPCodeSite,
    EnergyTraceSet,
)
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
    assert project.project_id is None

    project = Project(energy_trace_set, interventions, site, 'ABC')
    assert project.project_id == 'ABC'


def test_repr(energy_trace_set, interventions, site):
    project = Project(energy_trace_set, interventions, site)

    assert str(project).startswith("Project")
    assert "energy_trace_set=EnergyTraceSet" in str(project)
    assert "interventions=[]" in str(project)
    assert 'site=ZIPCodeSite("11111")' in str(project)

    project = Project(energy_trace_set, interventions, site, 'ABC')
    assert 'ABC' in str(project)
