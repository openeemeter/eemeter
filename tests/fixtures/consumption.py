import pytest

from eemeter.models import TemperatureSensitivityModel
from eemeter.consumption import Consumption,ConsumptionHistory
from eemeter.generator import ConsumptionGenerator
from eemeter.generator import generate_periods

from .weather import gsod_722880_2012_2014_weather_source

from datetime import datetime

@pytest.fixture
def consumption_history_1():
    c_list = [Consumption(687600000,"J","electricity",datetime(2012,9,26),datetime(2012,10,24)),
            Consumption(874800000,"J","electricity",datetime(2012,10,24),datetime(2012,11,21)),
            Consumption(1332000000,"J","electricity",datetime(2012,11,21),datetime(2012,12,27)),
            Consumption(1454400000,"J","electricity",datetime(2012,12,27),datetime(2013,1,29)),
            Consumption(1155600000,"J","electricity",datetime(2013,1,29),datetime(2013,2,26)),
            Consumption(1195200000,"J","electricity",datetime(2013,2,26),datetime(2013,3,27)),
            Consumption(1033200000,"J","electricity",datetime(2013,3,27),datetime(2013,4,25)),
            Consumption(752400000,"J","electricity",datetime(2013,4,25),datetime(2013,5,23)),
            Consumption(889200000,"J","electricity",datetime(2013,5,23),datetime(2013,6,22)),
            Consumption(3434400000,"J","electricity",datetime(2013,6,22),datetime(2013,7,26)),
            Consumption(828000000,"J","electricity",datetime(2013,7,26),datetime(2013,8,22)),
            Consumption(2217600000,"J","electricity",datetime(2013,8,22),datetime(2013,9,25)),
            Consumption(680400000,"J","electricity",datetime(2013,9,25),datetime(2013,10,23)),
            Consumption(1062000000,"J","electricity",datetime(2013,10,23),datetime(2013,11,22)),
            Consumption(1720800000,"J","electricity",datetime(2013,11,22),datetime(2013,12,27)),
            Consumption(1915200000,"J","electricity",datetime(2013,12,27),datetime(2014,1,30)),
            Consumption(1458000000,"J","electricity",datetime(2014,1,30),datetime(2014,2,27)),
            Consumption(1332000000,"J","electricity",datetime(2014,2,27),datetime(2014,3,29)),
            Consumption(954000000,"J","electricity",datetime(2014,3,29),datetime(2014,4,26)),
            Consumption(842400000,"J","electricity",datetime(2014,4,26),datetime(2014,5,28)),
            Consumption(1220400000,"J","electricity",datetime(2014,5,28),datetime(2014,6,25)),
            Consumption(1702800000,"J","electricity",datetime(2014,6,25),datetime(2014,7,25)),
            Consumption(1375200000,"J","electricity",datetime(2014,7,25),datetime(2014,8,23)),
            Consumption(1623600000,"J","electricity",datetime(2014,8,23),datetime(2014,9,25))]
    return ConsumptionHistory(c_list)

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
    start = datetime(2012,1,1)
    end = datetime(2014,12,31)
    periods = generate_periods(start,end,jitter_intensity=0)
    gen = ConsumptionGenerator("electricity", "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    return ConsumptionHistory(consumptions), model.param_dict_to_list(params)

@pytest.fixture(params=[([0, 1,65,1,75],1784.8507747107692),
                        ([10,2,61,1,73],5643.731382817317)])
def generated_consumption_history_with_annualized_usage_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    start = datetime(2012,1,1)
    end = datetime(2014,12,31)
    periods = generate_periods(start,end,jitter_intensity=0)
    gen = ConsumptionGenerator("electricity", "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    return ConsumptionHistory(consumptions), model.param_dict_to_list(params), request.param[1]

@pytest.fixture(params=[([0, 1,65,1,75],[0,.5,63,.7,75]),
                        ([10,2,61,1,73],[9, 1,61,.5,73])])
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

@pytest.fixture(params=[([-1, 1,60,1,70],True,929.0532551121628,0),
                        ([10,2,58,1,72],True,4927.478974253085,0),
                        ([0,2,64,3,66],True,6017.849018719649,0)])
def prism_outputs_1(request):
    model = TemperatureSensitivityModel(cooling=True,heating=True)
    params = {
        "base_consumption": request.param[0][0],
        "heating_slope": request.param[0][1],
        "heating_reference_temperature": request.param[0][2],
        "cooling_slope": request.param[0][3],
        "cooling_reference_temperature": request.param[0][4]
    }
    start = datetime(2012,1,1)
    end = datetime(2014,12,31)
    periods = generate_periods(start,end,jitter_intensity=0)
    gen = ConsumptionGenerator("electricity", "kWh", "degF", model, params)
    consumptions = gen.generate(gsod_722880_2012_2014_weather_source(), periods)
    return ConsumptionHistory(consumptions), model.param_dict_to_list(params), request.param[1], request.param[2], request.param[3]
