import pytest

from eemeter.models import TemperatureSensitivityModel
from eemeter.consumption import ConsumptionData
from eemeter.generator import MonthlyBillingConsumptionGenerator
from eemeter.generator import generate_monthly_billing_datetimes
from eemeter.evaluation import Period

from .weather import gsod_722880_2012_2014_weather_source

from scipy.stats import randint

from datetime import datetime

@pytest.fixture
def consumption_history_1():

    records = [
            {"start": datetime(2012,9,26), "value": 191},
            {"start": datetime(2012,10,24), "value": 243},
            {"start": datetime(2012,11,21), "value": 370},
            {"start": datetime(2012,12,27), "value": 404},
            {"start": datetime(2013,1,29), "value": 321},
            {"start": datetime(2013,2,26), "value": 332},
            {"start": datetime(2013,3,27), "value": 287},
            {"start": datetime(2013,4,25), "value": 209},
            {"start": datetime(2013,5,23), "value": 247},
            {"start": datetime(2013,6,22), "value": 954},
            {"start": datetime(2013,7,26), "value": 230},
            {"start": datetime(2013,8,22), "value": 616},
            {"start": datetime(2013,9,25), "value": 189},
            {"start": datetime(2013,10,23), "value": 295},
            {"start": datetime(2013,11,22), "value": 478},
            {"start": datetime(2013,12,27), "value": 532},
            {"start": datetime(2014,1,30), "value": 405},
            {"start": datetime(2014,2,27), "value": 370},
            {"start": datetime(2014,3,29), "value": 265},
            {"start": datetime(2014,4,26), "value": 234},
            {"start": datetime(2014,5,28), "value": 339},
            {"start": datetime(2014,6,25), "value": 473},
            {"start": datetime(2014,7,25), "value": 382},
            {"start": datetime(2014,8,23), "end": datetime(2014,9,25),
                "value": 451}]
    return ConsumptionData(records, "electricity", "kWh",
            record_type="arbitrary_start")

@pytest.fixture(params=[([10, 2, 61, 1, 73], "electricity", "kWh", "degF")])
def consumption_generator_1(request):
    model_params, fuel_type, consumption_unit_name, temperature_unit_name = \
            request.param
    model = TemperatureSensitivityModel(cooling=True, heating=True)
    params = {
        "base_consumption": model_params[0],
        "heating_slope": model_params[1],
        "heating_reference_temperature": model_params[2],
        "cooling_slope": model_params[3],
        "cooling_reference_temperature": model_params[4]
    }
    generator = MonthlyBillingConsumptionGenerator(fuel_type,
            consumption_unit_name, temperature_unit_name, model, params)
    return generator

@pytest.fixture(params=[([9, 1, 61, .5, 73], "electricity", "kWh", "degF")])
def consumption_generator_2(request):
    model_params, fuel_type, consumption_unit_name, temperature_unit_name = \
            request.param
    model = TemperatureSensitivityModel(cooling=True, heating=True)
    params = {
        "base_consumption": model_params[0],
        "heating_slope": model_params[1],
        "heating_reference_temperature": model_params[2],
        "cooling_slope": model_params[3],
        "cooling_reference_temperature": model_params[4]
    }
    generator = MonthlyBillingConsumptionGenerator(fuel_type,
            consumption_unit_name, temperature_unit_name, model, params)
    return generator

@pytest.fixture(params=[(Period(datetime(2012,1,1),datetime(2014,12,31)),randint(30,31))])
def generated_consumption_history_1(request,
        gsod_722880_2012_2014_weather_source, consumption_generator_1):
    period, dist = request.param
    datetimes = generate_monthly_billing_datetimes(period, dist)
    consumption_data = consumption_generator_1.generate(
            gsod_722880_2012_2014_weather_source, datetimes)
    param_list = model.param_dict_to_list(params)
    return consumption_data, param_list

@pytest.fixture(params=[(Period(datetime(2012,1,1),datetime(2014,12,31)),randint(30,31))])
def generated_consumption_history_2(request,
        gsod_722880_2012_2014_weather_source, consumption_generator_2):
    period, dist = request.param
    datetimes = generate_monthly_billing_datetimes(period, dist)
    consumption_data = consumption_generator_2.generate(
            gsod_722880_2012_2014_weather_source, datetimes)
    param_list = model.param_dict_to_list(params)
    return consumption_data, param_list

@pytest.fixture(params=[5643.731])
def generated_consumption_history_with_annualized_usage_1(request,
        generated_consumption_history_1):
    cd, param_list = generated_consumption_history_1
    annualized_usage = request.param[1]
    return cd, param_list, annualized_usage

@pytest.fixture
def generated_consumption_history_pre_post_1(request,
        generated_consumption_history_1, generated_consumption_history_2):
    cd_pre, param_list_pre = generated_consumption_history_1
    cd_post, param_list_post = generated_consumption_history_2
    record_type = "arbitrary"
    pre_records = cd_pre.records(record_type)
    post_records = cd_post.records(record_type)
    n_months_pre = len(pre_records) / 2
    all_records = pre_records[:n_months_pre] + \
            post_records[n_months_pre:]
    consumption_data = ConsumptionData(all_records, cd_pre.fuel_type,
            cd_post.unit_name, record_type="arbitrary")

    retrofit_date = post_records[n_months_pre]["start"]

    return consumption_data, param_list_pre, param_list_post, retrofit_date

@pytest.fixture(params=[1323.450])
def generated_consumption_history_pre_post_with_gross_savings_1(request,
        generated_consumption_history_pre_post_1):
    cd, param_list_pre, param_list_post, retrofit_date = \
            generated_consumption_history_pre_post_1
    gross_savings = param.request
    return cd, param_list_pre, param_list_post, retrofit_date, gross_savings

@pytest.fixture(params=[2020.733])
def generated_consumption_history_pre_post_with_annualized_gross_savings_1(
        request, generated_consumption_history_pre_post_1):
    cd, param_list_pre, param_list_post, retrofit_date = \
            generated_consumption_history_pre_post_1
    annualized_gross_savings = param.request
    return cd, param_list_pre, param_list_post, retrofit_date, \
        annualized_gross_savings

@pytest.fixture(params=[(Period(datetime(2012,1,1), datetime(2012,12,31)),360),
                        (Period(datetime(2012,1,1), datetime(2012,9,30)), 270),
                        (Period(datetime(2012,1,1), datetime(2012,7,1)), 180),
                        (Period(datetime(2012,1,1), datetime(2012,3,2)), 60),])
def time_span_1(request, consumption_generator_1,
        gsod_722880_2012_2014_weather_source):
    period, n_days = request.param
    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))
    consumption_data = consumption_generator_1.generate(
            gsod_722880_2012_2014_weather_source, datetimes)
    return consumption_data, n_days

@pytest.fixture(params=[
    ([0,1,65,1,75], Period(datetime(2012,1,1),datetime(2012,12,31)),
        1416.1000, 65, "degF"),
    ([10,2,61,1,73], Period(datetime(2012,1,1),datetime(2013,12,31)),
        2562.8000, 65, "degF"),
    ([8,1,16,1,22], Period(datetime(2012,1,1),datetime(2013,12,31)),
        1422.6478,18.33,"degC")
    ])
def generated_consumption_history_with_hdd_1(request,
        gsod_722880_2012_2014_weather_source):
    model_params, period, total_hdd, base, temp_unit = request.param

    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": model_params[0],
        "heating_slope": model_params[1],
        "heating_reference_temperature": model_params[2],
        "cooling_slope": model_params[3],
        "cooling_reference_temperature": model_params[4]
    }
    gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, params)

    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))
    consumption_data = gen.generate(gsod_722880_2012_2014_weather_source,
            periods)

    return consumption_data, total_hdd, base, temp_unit

@pytest.fixture(params=[
    ([0,1,65,1,75], Period(datetime(2012,1,1),datetime(2012,12,31)),
        1348.9000, 65, "degF"),
    ([10,2,61,1,73],Period(datetime(2012,1,1),datetime(2013,12,31)),
        3022.3000, 65, "degF"),
    ([8,1,16,1,22],Period(datetime(2012,1,1),datetime(2013,12,31)),
        1680.3254, 18.33, "degC")
    ])
def generated_consumption_history_with_cdd_1(request,
        gsod_722880_2012_2014_weather_source):
    model_params, period, total_cdd, base, temp_unit = request.param
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": model_params[0],
        "heating_slope": model_params[1],
        "heating_reference_temperature": model_params[2],
        "cooling_slope": model_params[3],
        "cooling_reference_temperature": model_params[4]
    }
    gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, params)

    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))
    consumption_data = gen.generate(gsod_722880_2012_2014_weather_source,
            periods)

    return consumption_data, total_cdd, base, temp_unit

@pytest.fixture(params=[
    ([0, 1,65,1,75], Period(datetime(2012,1,1),datetime(2012,12,31)),5,7,1),
    ([10,2,61,1,73], Period(datetime(2012,1,1),datetime(2013,12,31)),11,13,1)
    ])
def generated_consumption_history_with_n_periods_hdd_1(request,
        gsod_722880_2012_2014_weather_source):
    model_params, period, n_periods_1, n_periods_2, n_periods_3 = request.param
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": model_params[0],
        "heating_slope": model_params[1],
        "heating_reference_temperature": model_params[2],
        "cooling_slope": model_params[3],
        "cooling_reference_temperature": model_params[4]
    }
    gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, params)

    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))
    consumption_data = gen.generate(gsod_722880_2012_2014_weather_source,
            periods)
    return consumption_data, n_periods_1, n_periods_2, n_periods_3

@pytest.fixture(params=[
    ([0, 1,65,1,75], Period(datetime(2012,1,1),datetime(2012,12,31)),10,2,10),
    ([10,2,61,1,73], Period(datetime(2012,1,1),datetime(2014,12,31)),12,24,1),
    ([10,2,61,1,73], Period(datetime(2012,1,1),datetime(2012,12,27)),5,7,1),
    ([10,2,61,1,73], Period(datetime(2012,12,27),datetime(2014,12,31)),7,17,1)
    ])
def generated_consumption_history_with_n_periods_cdd_1(request,
        gsod_722880_2012_2014_weather_source):
    model_params, period, n_periods_1, n_periods_2, n_periods_3 = request.param
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": model_params[0],
        "heating_slope": model_params[1],
        "heating_reference_temperature": model_params[2],
        "cooling_slope": model_params[3],
        "cooling_reference_temperature": model_params[4]
    }
    gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, params)

    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))
    consumption_data = gen.generate(gsod_722880_2012_2014_weather_source,
            periods)
    return consumption_data, n_periods_1, n_periods_2, n_periods_3
