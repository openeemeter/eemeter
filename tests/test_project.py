from eemeter.project import Project
from eemeter.location import Location
from eemeter.evaluation import Period

from datetime import datetime

import pytest

def test_project():
    location = Location(zipcode="60604")
    baseline_period = Period(datetime(2014,1,1), datetime(2015,1,1))
    reporting_period = None
    project = Project(location, consumption=[],
            baseline_period=baseline_period,
            reporting_period=reporting_period,
            other_periods=[],
            weather_source=None,
            weather_normal_source=None)

    assert project.consumption == []
    assert project.baseline_period.start == datetime(2014,1,1)
    assert project.baseline_period.end == datetime(2015,1,1)
    assert project.reporting_period is None
    assert project.other_periods == []
    assert project.weather_source.station_id == "725340"
    assert project.weather_normal_source.station_id == "725340"
    assert len(project.all_periods()) == 1
    assert project._total_date_range()[0] == datetime(2014,1,1)
    assert project._total_date_range()[1] == datetime(2015,1,1)
    assert len(project.segmented_consumption_data()) == 0
