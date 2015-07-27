import pytest

from eemeter.models import TemperatureSensitivityModel
from eemeter.consumption import ConsumptionData
from eemeter.generator import MonthlyBillingConsumptionGenerator
from eemeter.generator import generate_monthly_billing_datetimes

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

@pytest.fixture(params=[[0, 1,65,1,75],
                        [10,2,61,1,73]])
def generated_consumption_history_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0],
        "heating_slope": request.param[1],
        "heating_reference_temperature": request.param[2],
        "cooling_slope": request.param[3],
        "cooling_reference_temperature": request.param[4]
    }
    period = Period(datetime(2012,1,1),datetime(2014,12,31))
    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))
    gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(),
            datetimes)
    return ConsumptionHistory(consumptions), model.param_dict_to_list(params)

@pytest.fixture(params=[([0, 1, 65, 1, 75],1784.8507747107692),
                        ([10, 2, 61, 1, 73],5643.731382817317)])
def generated_consumption_history_with_annualized_usage_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    period = Period(datetime(2012,1,1),datetime(2014,12,31))
    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))
    gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(),
            datetimes)
    return ConsumptionHistory(consumptions), model.param_dict_to_list(params),\
            request.param[1]

@pytest.fixture(params=[([0, 1, 65, 1, 75],[0, .5, 63, .7, 75]),
                        ([10, 2, 61, 1, 73],[9, 1, 61, .5, 73])])
def generated_consumption_history_pre_post_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    pre_params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    post_params = {
        "base_consumption": request.param[1][0],
        "heating_slope": request.param[1][1],
        "heating_reference_temperature": request.param[1][2],
        "cooling_slope": request.param[1][3],
        "cooling_reference_temperature": request.param[1][4]
    }
    pre_period = Period(datetime(2012,1,1),datetime(2013,6,15))
    post_period = Period(datetime(2013,6,15),datetime(2014,12,31))
    pre_datetimes = generate_monthly_billing_datetimes(pre_period,
            dist=randint(30,31))
    post_datetimes = generate_monthly_billing_datetimes(post_period,
            dist=randint(30,31))
    pre_gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, pre_params)
    post_gen = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model, post_params)
    pre_consumptions = pre_gen.generate(
            gsod_722880_2012_2014_weather_source(), pre_datetimes)
    post_consumptions = post_gen.generate(
            gsod_722880_2012_2014_weather_source(), post_datetimes)

    ch = ConsumptionHistory(pre_consumptions + post_consumptions)
    return ch, model.param_dict_to_list(pre_params), model.param_dict_to_list(post_params), retrofit

@pytest.fixture(params=[([0, 1,65,1,75],[0,.5,63,.7,75],641.7100012971271),
                        ([10,2,61,1,73],[9, 1,61,.5,73],1323.4500370841015)])
def generated_consumption_history_pre_post_with_gross_savings_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    pre_params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    post_params = {
        "base_consumption": request.param[1][0],
        "heating_slope": request.param[1][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[1][3],
        "cooling_reference_temperature": request.param[1][4]
    }
    start = datetime(2012,1,1)
    retrofit = datetime(2013,6,15)
    end = datetime(2014,12,31)
    pre_periods = generate_periods(start,retrofit,jitter_intensity=0)
    post_periods = generate_periods(retrofit,end,jitter_intensity=0)
    pre_gen = ConsumptionGenerator("electricity", "kWh", "degF", model, pre_params)
    post_gen = ConsumptionGenerator("electricity", "kWh", "degF", model, post_params)
    pre_consumptions = pre_gen.generate(gsod_722880_2012_2014_weather_source(), pre_periods)
    post_consumptions = post_gen.generate(gsod_722880_2012_2014_weather_source(), post_periods)
    ch = ConsumptionHistory(pre_consumptions + post_consumptions)
    return ch, model.param_dict_to_list(pre_params), model.param_dict_to_list(post_params), retrofit, request.param[2]

@pytest.fixture(params=[([0, 1,65,1,75],[0,.5,63,.7,75],1545.2814557846802),
                        ([10,2,61,1,73],[9, 1,61,.5,73],2020.7330084855778)])
def generated_consumption_history_pre_post_with_annualized_gross_savings_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    pre_params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    post_params = {
        "base_consumption": request.param[1][0],
        "heating_slope": request.param[1][1],
        "heating_reference_temperature": request.param[1][2],
        "cooling_slope": request.param[1][3],
        "cooling_reference_temperature": request.param[1][4]
    }
    start = datetime(2012,1,1)
    retrofit = datetime(2013,6,15)
    end = datetime(2014,12,31)
    pre_periods = generate_periods(start,retrofit,jitter_intensity=0)
    post_periods = generate_periods(retrofit,end,jitter_intensity=0)
    pre_gen = ConsumptionGenerator("electricity", "kWh", "degF", model, pre_params)
    post_gen = ConsumptionGenerator("electricity", "kWh", "degF", model, post_params)
    pre_consumptions = pre_gen.generate(gsod_722880_2012_2014_weather_source(), pre_periods)
    post_consumptions = post_gen.generate(gsod_722880_2012_2014_weather_source(), post_periods)
    ch = ConsumptionHistory(pre_consumptions + post_consumptions)
    return ch, model.param_dict_to_list(pre_params), model.param_dict_to_list(post_params), retrofit, request.param[2]

@pytest.fixture(params=[([10,2,58,1,72],"electricity",(datetime(2012,1,1),datetime(2012,12,31)),360),
                        ([10,2,58,1,72],"electricity",(datetime(2012,1,1),datetime(2012,9,30)),270),
                        ([10,2,58,1,72],"electricity",(datetime(2012,1,1),datetime(2012,7,1)),180),
                        ([10,2,58,1,72],"electricity",(datetime(2012,1,1),datetime(2012,3,2)),60),])
def time_span_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    fuel_type = request.param[1]
    start, end = request.param[2]
    n_days = request.param[3]
    periods = generate_periods(start,end,jitter_intensity=0)
    gen = ConsumptionGenerator(fuel_type, "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    param_list = model.param_dict_to_list(params)
    return ConsumptionHistory(consumptions), fuel_type, n_days

@pytest.fixture(params=[([0, 1,65,1,75],(datetime(2012,1,1),datetime(2012,12,31)),1416.100000000004,65,"degF"),
                        ([10,2,61,1,73],(datetime(2012,1,1),datetime(2013,12,31)),2562.8000000000056,65,"degF"),
                        ([8, 1,16,1,22],(datetime(2012,1,1),datetime(2013,12,31)),1422.6478,18.33,"degC"),
                        ])
def generated_consumption_history_with_hdd_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    start,end = request.param[1]
    periods = generate_periods(start,end,jitter_intensity=0)
    fuel_type = "electricity"
    gen = ConsumptionGenerator(fuel_type, "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    return ConsumptionHistory(consumptions),fuel_type, request.param[2], request.param[3], request.param[4]

@pytest.fixture(params=[([0, 1,65,1,75],(datetime(2012,1,1),datetime(2012,12,31)),1348.8999999999976,65,"degF"),
                        ([10,2,61,1,73],(datetime(2012,1,1),datetime(2013,12,31)),3022.2999999999947,65,"degF"),
                        ([8, 1,16,1,22],(datetime(2012,1,1),datetime(2013,12,31)),1680.3254,18.33,"degC"),
                        ])
def generated_consumption_history_with_cdd_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    start,end = request.param[1]
    periods = generate_periods(start,end,jitter_intensity=0)
    fuel_type = "electricity"
    gen = ConsumptionGenerator(fuel_type, "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    return ConsumptionHistory(consumptions), fuel_type, request.param[2], request.param[3], request.param[4]

@pytest.fixture(params=[([0, 1,65,1,75],(datetime(2012,1,1),datetime(2012,12,31)),5,7,1),
                        ([10,2,61,1,73],(datetime(2012,1,1),datetime(2013,12,31)),11,13,1)])
def generated_consumption_history_with_n_periods_hdd_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    start,end = request.param[1]
    periods = generate_periods(start,end,jitter_intensity=0)
    fuel_type = "electricity"
    gen = ConsumptionGenerator(fuel_type, "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    return ConsumptionHistory(consumptions), fuel_type, request.param[2], request.param[3], request.param[4]

@pytest.fixture(params=[([0, 1,65,1,75],(datetime(2012,1,1),datetime(2012,12,31)),10,2,10),
                        ([10,2,61,1,73],(datetime(2012,1,1),datetime(2014,12,31)),12,24,1),
                        ([10,2,61,1,73],(datetime(2012,1,1),datetime(2012,12,27)),5,7,1),
                        ([10,2,61,1,73],(datetime(2012,12,27),datetime(2014,12,31)),7,17,1)])
def generated_consumption_history_with_n_periods_cdd_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    start,end = request.param[1]
    periods = generate_periods(start,end,jitter_intensity=0)
    fuel_type = "electricity"
    gen = ConsumptionGenerator(fuel_type, "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    return ConsumptionHistory(consumptions), fuel_type, request.param[2], request.param[3], request.param[4 ]
