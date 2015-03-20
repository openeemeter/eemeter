from eemeter.meter import PRISMMeter

from fixtures.weather import gsod_722880_2012_2014_weather_source
from fixtures.weather import tmy3_722880_weather_source

from fixtures.consumption import prism_outputs_1

from numpy.testing import assert_allclose

RTOL = 1e-2
ATOL = 1e-2

import pytest

@pytest.mark.slow
def test_princeton_scorekeeping_method(prism_outputs_1,
                                       gsod_722880_2012_2014_weather_source,
                                       tmy3_722880_weather_source):
    meter = PRISMMeter()

    ch, elec_params, elec_presence, elec_annualized_usage, elec_error = prism_outputs_1
    result = meter.evaluate(consumption_history=ch,
                            weather_source=gsod_722880_2012_2014_weather_source,
                            weather_normal_source=tmy3_722880_weather_source)



    assert result.get("electricity_presence") == elec_presence
    assert_allclose(result.get("temp_sensitivity_params_electricity"),
            elec_params, rtol=RTOL, atol=ATOL)
    assert_allclose(result.get("annualized_usage_electricity"),
            elec_annualized_usage, rtol=RTOL, atol=ATOL)
    assert_allclose(result.get("daily_standard_error_electricity"),
            elec_error, rtol=RTOL, atol=ATOL)

    assert not result.get("natural_gas_presence")
    assert result.get("temp_sensitivity_params_natural_gas") is None
    assert result.get("annualized_usage_natural_gas") is None
    assert result.get("daily_standard_error_natural_gas") is None

