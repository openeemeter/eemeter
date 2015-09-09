from eemeter.location import Location
from eemeter.evaluation import Period
from eemeter.weather import GSODWeatherSource
from eemeter.generator import MonthlyBillingConsumptionGenerator
from eemeter.generator import generate_monthly_billing_datetimes
from eemeter.consumption import ConsumptionData
from eemeter.models import AverageDailyTemperatureSensitivityModel
from eemeter.project import Project

from scipy.stats import randint

from datetime import datetime
import pytz


def get_example_project(zipcode):

    # location
    location = Location(zipcode=zipcode)
    station = location.station
    weather_source = GSODWeatherSource(station,2011,2015)

    # model
    model_e = AverageDailyTemperatureSensitivityModel(cooling=True, heating=True)
    model_g = AverageDailyTemperatureSensitivityModel(cooling=False, heating=True)

    # model params
    params_e_b = {
        "cooling_slope": 1,
        "heating_slope": 1,
        "base_daily_consumption": 30,
        "cooling_balance_temperature": 73,
        "heating_balance_temperature": 68,
    }
    params_e_r = {
        "cooling_slope": .5,
        "heating_slope": .5,
        "base_daily_consumption": 15,
        "cooling_balance_temperature": 73,
        "heating_balance_temperature": 68,
    }
    params_g_b = {
        "heating_slope": .2,
        "base_daily_consumption": 2,
        "heating_balance_temperature": 68,
    }
    params_g_r = {
        "heating_slope": .1,
        "base_daily_consumption": 1,
        "heating_balance_temperature": 68,
    }

    #generators
    gen_e_b = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model_e, params_e_b)
    gen_e_r = MonthlyBillingConsumptionGenerator("electricity", "kWh", "degF",
            model_e, params_e_r)
    gen_g_b = MonthlyBillingConsumptionGenerator("natural_gas", "therm", "degF",
            model_g, params_g_b)
    gen_g_r = MonthlyBillingConsumptionGenerator("natural_gas", "therm", "degF",
            model_g, params_g_r)

    # time periods
    period = Period(datetime(2011,1,1,tzinfo=pytz.utc), datetime(2015,1,1,tzinfo=pytz.utc))
    datetimes = generate_monthly_billing_datetimes(period, dist=randint(30,31))

    # consumption data
    cd_e_b = gen_e_b.generate(weather_source, datetimes, daily_noise_dist=None)
    cd_e_r = gen_e_r.generate(weather_source, datetimes, daily_noise_dist=None)
    cd_g_b = gen_g_b.generate(weather_source, datetimes, daily_noise_dist=None)
    cd_g_r = gen_g_r.generate(weather_source, datetimes, daily_noise_dist=None)

    # periods
    periods = cd_e_b.periods()
    reporting_period = Period(datetime(2013,1,1,tzinfo=pytz.utc), datetime(2015,1,1,tzinfo=pytz.utc))
    baseline_period = Period(datetime(2011,1,1,tzinfo=pytz.utc), datetime(2013,1,1,tzinfo=pytz.utc))

    # records
    records_e = []
    records_g = []
    for e_b, e_r, g_b, g_r, p in zip(cd_e_b.data, cd_e_r.data, cd_g_b.data, cd_g_r.data, periods):
        e = e_r if p in reporting_period else e_b
        g = g_r if p in reporting_period else g_b
        record_e = {"start": p.start, "end": p.end, "value": e}
        record_g = {"start": p.start, "end": p.end, "value": g}
        records_e.append(record_e)
        records_g.append(record_g)

    # consumption_data
    cd_e = ConsumptionData(records_e, "electricity", "kWh",
            record_type="arbitrary")
    cd_g = ConsumptionData(records_g, "natural_gas", "therm",
            record_type="arbitrary")
    consumptions = [cd_e, cd_g]

    # project
    project = Project(location, consumptions, baseline_period, reporting_period)

    return project
