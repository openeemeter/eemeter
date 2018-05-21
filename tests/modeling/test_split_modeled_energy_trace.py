import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytest
import pytz

from eemeter.modeling.formatters import ModelDataFormatter
from eemeter.modeling.models.seasonal import SeasonalElasticNetCVModel
from eemeter.modeling.split import SplitModeledEnergyTrace
from eemeter.structures import (
    EnergyTrace,
    ModelingPeriod,
    ModelingPeriodSet,
)
from eemeter.weather import WeatherSource


def _fake_temps(usaf_id, start, end, normalized, use_cz2010):
    # sinusoidal fake temperatures in degC
    dates = pd.date_range(start, end, freq='H', tz=pytz.UTC)
    num_years = end.year - start.year + 1
    n = dates.shape[0]
    avg_temp = 15
    temp_range = 15
    period_offset = - (2 * np.pi / 3)
    temp_offsets = np.sin(
        (2 * np.pi * num_years * np.arange(n) / n) + period_offset)
    temps = avg_temp + (temp_range * temp_offsets)
    return pd.Series(temps, index=dates, dtype=float)


@pytest.fixture
def monkeypatch_temperature_data(monkeypatch):
    monkeypatch.setattr(
        'eemeter.weather.eeweather_wrapper._get_temperature_data_eeweather',
        _fake_temps
    )


@pytest.fixture
def trace():
    data = {
        "value": np.tile(1, (365,)),
        "estimated": np.tile(False, (365,)),
    }
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=365, freq='D', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def modeling_period_set():
    modeling_period_1 = ModelingPeriod(
        "BASELINE",
        end_date=datetime(2000, 9, 1, tzinfo=pytz.UTC),
    )
    modeling_period_2 = ModelingPeriod(
        "REPORTING",
        start_date=datetime(2001, 1, 1, tzinfo=pytz.UTC),
    )
    modeling_periods = {
        "modeling_period_1": modeling_period_1,
        "modeling_period_2": modeling_period_2,
    }

    grouping = [
        ("modeling_period_1", "modeling_period_2"),
    ]

    return ModelingPeriodSet(modeling_periods, grouping)


def test_basic_usage(trace, modeling_period_set, monkeypatch_temperature_data):

    # create SplitModeledEnergyTrace
    formatter = ModelDataFormatter('D')
    model_mapping = {
        'modeling_period_1': SeasonalElasticNetCVModel(65, 65),
        'modeling_period_2': SeasonalElasticNetCVModel(65, 65),
    }
    smet = SplitModeledEnergyTrace(
        trace, formatter, model_mapping, modeling_period_set)

    # fit normally
    outputs = smet.fit()
    assert 'modeling_period_1' in smet.fit_outputs
    assert 'modeling_period_2' in smet.fit_outputs
    assert len(smet.fit_outputs) == 2
    assert outputs['modeling_period_1']['status'] == 'SUCCESS'
    assert outputs['modeling_period_1']['start_date'] == \
        datetime(2000, 1, 1, tzinfo=pytz.UTC)
    assert outputs['modeling_period_1']['end_date'] == \
        datetime(2000, 9, 1, tzinfo=pytz.UTC)
    assert outputs['modeling_period_1']['n_rows'] == 245

    index = pd.date_range('2001-01-01', periods=6, freq='D', tz=pytz.UTC)

    demand_fixture_data = \
        smet.formatter.create_demand_fixture(index, mock_isd_weather_source)

    mp1_pred, variance = smet.predict(
        'modeling_period_1', demand_fixture_data, summed=False)
    mp2_pred = smet.predict('modeling_period_2', demand_fixture_data)

    assert mp1_pred.shape == (6,)
    assert mp2_pred is None
    assert variance > 0

    with pytest.raises(KeyError):
        smet.predict('modeling_period_3', demand_fixture_data)

    def callable_(formatter, model, returnme):
        return returnme

    mp1_deriv = smet.compute_derivative(
        'modeling_period_1', callable_, {"returnme": "A"})
    mp2_deriv = smet.compute_derivative(
        'modeling_period_2', callable_, {"returnme": "A"})

    assert mp1_deriv == "A"
    assert mp2_deriv is None
    pred, variance = smet.predict(
        'modeling_period_1', demand_fixture_data, summed=True)

    # predict summed
    assert_allclose(pred, 6.035919)
    assert variance > 0

    # bad weather source
    smet.fit(None)
    assert outputs['modeling_period_1']['status'] == 'FAILURE'

def test_bad_weather_source(trace, modeling_period_set):

    # create SplitModeledEnergyTrace
    formatter = ModelDataFormatter('D')
    model_mapping = {
        'modeling_period_1': SeasonalElasticNetCVModel(65, 65),
        'modeling_period_2': SeasonalElasticNetCVModel(65, 65),
    }
    smet = SplitModeledEnergyTrace(
        trace, formatter, model_mapping, modeling_period_set)

    # need to see that it gives a data sufficiency exception
    outputs = smet.fit(None)
    assert 'DataSufficiencyException' in outputs['modeling_period_1']['traceback']
    assert 'DataSufficiencyException' in outputs['modeling_period_2']['traceback']
