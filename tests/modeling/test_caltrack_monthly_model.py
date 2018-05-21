import tempfile

from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
import pytz

from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter
)
from eemeter.modeling.exceptions import (
    DataSufficiencyException,
)
from eemeter.structures import EnergyTrace
from eemeter.modeling.models import CaltrackMonthlyModel
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
def billing_trace():
    data = {
        "value": [1, 1, 1, 1, np.nan] + [1,]*13,
        "estimated": [False, False, True, False, False] + [False,]*13
    }
    columns = ["value", "estimated"]
    index = [
        datetime(2011, 1, 1, tzinfo=pytz.UTC),
        datetime(2011, 2, 1, tzinfo=pytz.UTC),
        datetime(2011, 3, 2, tzinfo=pytz.UTC),
        datetime(2011, 4, 3, tzinfo=pytz.UTC),
        datetime(2011, 4, 29, tzinfo=pytz.UTC),
    ] + [
        datetime(2011, 6, 1, tzinfo=pytz.UTC) + timedelta(days=30*i)
        for i in range(13)
    ]
    df = pd.DataFrame(data, index=index, columns=columns)
    return EnergyTrace("ELECTRICITY_CONSUMPTION_SUPPLIED", df, unit="KWH")


@pytest.fixture
def input_df(monkeypatch_temperature_data, daily_trace,
    mock_isd_weather_source):
    mdf = ModelDataFormatter("D")
    return mdf.create_input(daily_trace, mock_isd_weather_source)


@pytest.fixture
def input_billing_df(monkeypatch_temperature_data, billing_trace,
    mock_isd_weather_source):
    mdbf = ModelDataBillingFormatter()
    return mdbf.create_input(billing_trace, mock_isd_weather_source)

def test_sufficiency_criteria():
    m_baseline = CaltrackMonthlyModel(
        fit_cdd=True, modeling_period_interpretation='baseline')
    m_reporting = CaltrackMonthlyModel(
        fit_cdd=True, modeling_period_interpretation='reporting')

    # too short
    too_short_df = pd.DataFrame({
        'upd': [1 for _ in range(5)],
        'HDD_XX': [1 for _ in range(5)]
    })

    with pytest.raises(DataSufficiencyException) as e:
        m_baseline.meets_sufficiency_or_error(too_short_df)
    message = str(e.value)
    assert message == (
        'Data does not meet minimum contiguous months'
        ' requirement. The last 12 months of a baseline period must'
        ' have non-NaN energy and temperature values. In this case,'
        ' there were only 5 months in the series.'
    )

    with pytest.raises(DataSufficiencyException) as e:
        m_reporting.meets_sufficiency_or_error(too_short_df)
    message = str(e.value)
    assert message == (
        'Data does not meet minimum contiguous months'
        ' requirement. The first 12 months of a reporting period must'
        ' have non-NaN energy and temperature values. In this case,'
        ' there were only 5 months in the series.'
    )

    # no error
    upd_ok_temp_ok_baseline = pd.DataFrame({
        'upd': [1 if i > 3 else np.nan for i in range(20)],
        'HDD_XX': [1 if i > 3 else np.nan for i in range(20)],
    })
    m_baseline.meets_sufficiency_or_error(upd_ok_temp_ok_baseline)

    upd_bad_temp_bad_reporting = pd.DataFrame({
        'upd': [1 if i > 3 else np.nan for i in range(20)],
        'HDD_XX': [1 if i > 3 else np.nan for i in range(20)],
    })
    with pytest.raises(DataSufficiencyException) as e:
        m_reporting.meets_sufficiency_or_error(upd_bad_temp_bad_reporting)
    message = str(e.value)
    assert message == (
        'Data does not meet minimum contiguous months'
        ' requirement. The first 12 months of a reporting period must have'
        ' at least 15 valid days of energy and temperature data. In this case,'
        ' only 9 and 9 of the first 12 months of energy and temperature data'
        ' met that requirement, respectively.'
    )

    upd_bad_temp_ok_baseline = pd.DataFrame({
        'upd': [1 if i > 17 else np.nan for i in range(20)],
        'HDD_XX': [1 if i > 3 else np.nan for i in range(20)],
    })
    with pytest.raises(DataSufficiencyException) as e:
        m_baseline.meets_sufficiency_or_error(upd_bad_temp_ok_baseline)
    message = str(e.value)
    assert message == (
        'Data does not meet minimum contiguous months'
        ' requirement. The last 12 months of a baseline period must have at'
        ' least 15 valid days of energy and temperature data. In this case,'
        ' only 2 of the last 12 months of energy data met that requirement.'
    )

    upd_ok_temp_bad_reporting = pd.DataFrame({
        'upd': [np.nan if i > 17 else 1 for i in range(20)],
        'HDD_XX': [np.nan if i > 3 else 1 for i in range(20)],
    })
    with pytest.raises(DataSufficiencyException) as e:
        m_reporting.meets_sufficiency_or_error(upd_ok_temp_bad_reporting)
    message = str(e.value)
    assert message == (
        'Data does not meet minimum contiguous months requirement. The first'
        ' 12 months of a reporting period must have at least 15 valid days of'
        ' energy and temperature data. In this case, only 4 of the first 12'
        ' months of temperature data met that requirement.'
    )

    zeros = pd.DataFrame({
        'upd': [0 for i in range(20)],
        'HDD_XX': [0 for i in range(20)],
    })
    with pytest.raises(DataSufficiencyException) as e:
        m_reporting.meets_sufficiency_or_error(zeros)
    message = str(e.value)
    assert message == (
        'Energy trace data is all or nearly all zero'
    )

def test_fit_cdd(input_df):
    m = CaltrackMonthlyModel(fit_cdd=True)
    assert str(m).startswith("Caltrack")
    assert m.n is None
    assert m.params is None
    assert m.r2 is None
    assert m.rmse is None
    assert m.y is None

    output = m.fit(input_df)

    assert "r2" in output
    assert "rmse" in output
    assert "cvrmse" in output
    assert "model_params" in output
    assert "n" in output

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0., rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, variance = m.predict(input_df, summed=False)

    assert predict.shape == (365,)
    assert_allclose(predict[datetime(2000, 1, 1, tzinfo=pytz.UTC)], 1.)
    assert all(variance > 0)

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0.00000000000, rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, variance = m.predict(input_df)

    assert_allclose(predict, 365.)
    assert variance > 0


def test_fit_cdd_false(input_df):
    m = CaltrackMonthlyModel(fit_cdd=False)
    assert str(m).startswith("Caltrack")
    assert m.n is None
    assert m.params is None
    assert m.r2 is None
    assert m.rmse is None
    assert m.y is None

    output = m.fit(input_df)

    assert "r2" in output
    assert "rmse" in output
    assert "cvrmse" in output
    assert "model_params" in output
    assert "n" in output

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0., rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, variance = m.predict(input_df, summed=False)

    assert predict.shape == (365,)
    assert_allclose(predict[datetime(2000, 1, 1, tzinfo=pytz.UTC)], 1.)
    assert all(variance > 0)

    assert m.n == 12
    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params
    assert m.r2 == 0.0
    assert_allclose(m.rmse, 0.00000000000, rtol=1e-5, atol=1e-5)
    assert m.y.shape == (12, 1)

    predict, variance = m.predict(input_df)

    assert_allclose(predict, 365.)
    assert variance > 0


def test_basic_billing(input_billing_df, monkeypatch_temperature_data,
    mock_isd_weather_source):
    m = CaltrackMonthlyModel(fit_cdd=True)
    assert str(m).startswith("Caltrack")
    assert m.n is None
    assert m.params is None
    assert m.r2 is None
    assert m.rmse is None
    assert m.y is None

    output = m.fit(input_billing_df)

    assert "r2" in output
    assert "rmse" in output
    assert "cvrmse" in output
    assert "model_params" in output
    assert "n" in output

    assert 'formula' in m.params
    assert 'X_design_info' in m.params
    assert 'coefficients' in m.params

    index = pd.date_range('2011-01-01', freq='D', periods=365, tz=pytz.UTC)
    formatter = ModelDataBillingFormatter()
    formatted_predict_data = formatter.create_demand_fixture(index,
        mock_isd_weather_source)

    outputs, variance = m.predict(formatted_predict_data, summed=False)
    assert outputs.shape == (365,)
    assert all(variance > 0)

    outputs, variance = m.predict(formatted_predict_data, summed=True)
    assert outputs > 0
    assert variance > 0
