from datetime import datetime
import tempfile
import json

import numpy as np
import pandas as pd
import pytest
import pytz

from eemeter.io.serializers import serialize_split_modeled_energy_trace
from eemeter.structures import (
    EnergyTrace,
    ModelingPeriod,
    ModelingPeriodSet,
)
from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter,
)
from eemeter.modeling.models import (
    SeasonalElasticNetCVModel,
    BillingElasticNetCVModel,
)
from eemeter.modeling.split import SplitModeledEnergyTrace
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
def mock_isd_weather_source():
    ws = WeatherSource('722880', False, False)
    return ws


@pytest.fixture
def daily_trace():
    data = {
        "value": np.tile(1, (365,)),
        "estimated": np.tile(False, (365,)),
    }
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=365, freq='D', tz=pytz.UTC)
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def monthly_trace():
    data = {
        "value": np.tile(1, (24,)),
        "estimated": np.tile(False, (24,)),
    }
    columns = ["value", "estimated"]
    index = pd.date_range('2000-01-01', periods=24, freq='MS', tz=pytz.UTC)
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


@pytest.fixture
def split_modeled_energy_trace_daily(daily_trace, modeling_period_set,
    monkeypatch_temperature_data, mock_isd_weather_source):

    # create SplitModeledEnergyTrace
    formatter = ModelDataFormatter('D')
    model_mapping = {
        'modeling_period_1': SeasonalElasticNetCVModel(65, 65),
        'modeling_period_2': SeasonalElasticNetCVModel(65, 65),
    }
    smet = SplitModeledEnergyTrace(
        daily_trace, formatter, model_mapping, modeling_period_set)

    smet.fit(mock_isd_weather_source)
    return smet


@pytest.fixture
def split_modeled_energy_trace_monthly(monthly_trace, modeling_period_set,
    monkeypatch_temperature_data, mock_isd_weather_source):

    # create SplitModeledEnergyTrace
    formatter = ModelDataBillingFormatter()
    model_mapping = {
        'modeling_period_1': BillingElasticNetCVModel(65, 65),
        'modeling_period_2': BillingElasticNetCVModel(65, 65),
    }
    smet = SplitModeledEnergyTrace(
        monthly_trace, formatter, model_mapping, modeling_period_set)

    smet.fit(mock_isd_weather_source)
    return smet


def test_basic_usage_daily(split_modeled_energy_trace_daily):
    serialized = serialize_split_modeled_energy_trace(
        split_modeled_energy_trace_daily)

    # no error
    json.dumps(serialized)

    type_ = serialized["type"]
    assert type_ == "SPLIT_MODELED_ENERGY_TRACE"

    fits = serialized["fits"]
    mp1 = fits["modeling_period_1"]
    mp2 = fits["modeling_period_2"]

    assert mp1['status'] == "SUCCESS"
    assert mp1['traceback'] is None
    assert mp1['input_data'] is not None
    assert mp1['start_date'] is not None
    assert mp1['end_date'] is not None
    assert mp1['n_rows'] is not None

    model_fit = mp1['model_fit']
    assert model_fit["r2"] is not None
    assert model_fit["cvrmse"] is not None
    assert model_fit["rmse"] is not None
    assert model_fit["lower"] is not None
    assert model_fit["upper"] is not None
    assert model_fit["n"] is not None
    assert model_fit["model_params"] is not None

    assert mp2['status'] == "FAILURE"
    assert mp2['traceback'] is not None
    assert len(mp2['input_data']) == 0
    assert mp2['start_date'] is None
    assert mp2['end_date'] is None
    assert mp2['n_rows'] is not None
    assert mp2['model_fit']['r2'] is None
    assert mp2['model_fit']['cvrmse'] is None
    assert mp2['model_fit']['rmse'] is None
    assert mp2['model_fit']['lower'] is None
    assert mp2['model_fit']['upper'] is None
    assert mp2['model_fit']['n'] is None
    assert mp2['model_fit']['model_params'] is not None

    modeling_period_set = serialized["modeling_period_set"]
    modeling_periods = modeling_period_set["modeling_periods"]

    modeling_period_groups = modeling_period_set["modeling_period_groups"]
    assert len(modeling_period_groups) == 1
    assert modeling_period_groups[0]["baseline"] == "modeling_period_1"
    assert modeling_period_groups[0]["reporting"] == "modeling_period_2"

    assert modeling_periods["modeling_period_1"]["end_date"] == \
        '2000-09-01T00:00:00+00:00'


def test_basic_usage_monthly(split_modeled_energy_trace_monthly):
    serialized = serialize_split_modeled_energy_trace(
        split_modeled_energy_trace_monthly)

    # no error
    json.dumps(serialized)

    fits = serialized["fits"]
    mp1 = fits["modeling_period_1"]
    mp2 = fits["modeling_period_2"]

    assert mp1['status'] == "SUCCESS"
    assert mp1['traceback'] is None
    assert mp1['input_data'] is not None
    assert mp1['start_date'] is not None
    assert mp1['end_date'] is not None
    assert mp1['n_rows'] is not None

    model_fit = mp1['model_fit']
    assert model_fit["r2"] is not None
    assert model_fit["cvrmse"] is not None
    assert model_fit["rmse"] is not None
    assert model_fit["lower"] is not None
    assert model_fit["upper"] is not None
    assert model_fit["n"] is not None
    assert model_fit["model_params"] is not None

    assert mp2['status'] == "SUCCESS"
    assert mp2['traceback'] is None
    assert len(mp2['input_data']) > 0
    assert mp2['start_date'] is not None
    assert mp2['end_date'] is not None
    assert mp2['n_rows'] is not None
    assert mp2['model_fit'] is not None

    modeling_period_set = serialized["modeling_period_set"]

    modeling_period_groups = modeling_period_set["modeling_period_groups"]
    assert len(modeling_period_groups) == 1
    assert modeling_period_groups[0]["baseline"] == "modeling_period_1"
    assert modeling_period_groups[0]["reporting"] == "modeling_period_2"

    modeling_periods = modeling_period_set["modeling_periods"]

    assert modeling_periods["modeling_period_1"]["end_date"] == \
        '2000-09-01T00:00:00+00:00'
