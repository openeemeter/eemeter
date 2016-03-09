from eemeter.consumption import ConsumptionData
from eemeter.examples import get_example_project
from eemeter.meter import DefaultResidentialMeter
from eemeter.meter import DataCollection


def test_tutorial():
    project = get_example_project("94087")
    meter = DefaultResidentialMeter()
    results = meter.evaluate(DataCollection(project=project))

    electricity_usage_pre = results.get_data("annualized_usage", ["electricity", "baseline"]).value
    electricity_usage_post = results.get_data("annualized_usage", ["electricity", "reporting"]).value
    natural_gas_usage_pre = results.get_data("annualized_usage", ["natural_gas", "baseline"]).value
    natural_gas_usage_post = results.get_data("annualized_usage", ["natural_gas", "reporting"]).value

    electricity_savings = (electricity_usage_pre - electricity_usage_post) / electricity_usage_pre
    natural_gas_savings = (natural_gas_usage_pre - natural_gas_usage_post) / natural_gas_usage_pre

    json_data = results.json()
    assert "consumption" in json_data
    assert "weather_source" in json_data
    assert "gross_savings" in json_data
