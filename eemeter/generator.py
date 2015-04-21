from datetime import timedelta
from eemeter.consumption import Consumption
from eemeter.consumption import DatetimePeriod

from eemeter.meter import AnnualizedUsageMeter

import random
import numpy as np

class ConsumptionGenerator:
    """Class for generating consumptions given a particular model, parameters,
    and weather data. Useful for testing meters or meter deployments before
    observed project data becomes available.
    """
    def __init__(self, fuel_type, consumption_unit_name, temperature_unit_name,
            model, params):
        self.fuel_type = fuel_type
        self.consumption_unit_name = consumption_unit_name
        self.temperature_unit_name = temperature_unit_name

        self.model = model
        self.params = self.model.param_dict_to_list(params)

    def generate(self, weather_source, periods, noise=None):
        """Returns a list of consumption instances given a particular weather
        source and a list of DatetimePeriod instances.

        `noise` should be an instance of scipy.stats.rv_continuous,
        e.g. scipy.stats.normal(). Noise is additive and sampled independently
        for each period.
        """
        period_daily_temps = weather_source.daily_temperatures(periods,self.temperature_unit_name)

        usages = self.model.compute_usage_estimates(self.params,period_daily_temps)

        consumptions = []
        for u,period in zip(usages,periods):
            if noise is not None:
                u += np.sum(noise.rvs(size=period.timedelta.days))

            c = Consumption(u, self.consumption_unit_name, self.fuel_type, period.start, period.end)
            consumptions.append(c)

        return consumptions

class ProjectGenerator:
    """Class for generating complete projects given a particular parameter
    distributions and weather data. Useful for testing meters or meter
    deployments before observed project data becomes available.
    """
    def __init__(self, electricity_model, gas_model,
            electricity_param_distributions, electricity_param_delta_distributions,
            gas_param_distributions, gas_param_delta_distributions,
            temperature_unit_name="degF"):

        self.electricity_model = electricity_model
        self.elec_param_dists = electricity_param_distributions
        self.elec_param_delta_dists = electricity_param_delta_distributions

        self.gas_model = gas_model
        self.gas_param_dists = gas_param_distributions
        self.gas_param_delta_dists = gas_param_delta_distributions

        self.temperature_unit_name = temperature_unit_name

    def generate(self, weather_source, weather_normal_source, electricity_periods, gas_periods,
            retrofit_start_date, retrofit_completion_date,
            electricity_noise=None,gas_noise=None):
        """Returns a simple simulated project consisting of generated
        electricity and gas consumptions and estimated savings for each.
        """

        electricity_consumptions, estimated_electricity_savings, elec_pre_params, elec_post_params = self._generate_fuel_consumptions(
                weather_source, weather_normal_source, electricity_periods,
                self.electricity_model, self.elec_param_dists,
                self.elec_param_delta_dists, electricity_noise,
                retrofit_start_date, retrofit_completion_date,
                "electricity", "kWh", self.temperature_unit_name)

        gas_consumptions, estimated_gas_savings, gas_pre_params, gas_post_params = self._generate_fuel_consumptions(
                weather_source, weather_normal_source, gas_periods,
                self.gas_model, self.gas_param_dists,
                self.gas_param_delta_dists, gas_noise,
                retrofit_start_date, retrofit_completion_date,
                "natural_gas", "therm", self.temperature_unit_name)

        results = {
            "electricity_consumptions": electricity_consumptions,
            "natural_gas_consumptions": gas_consumptions,
            "electricity_estimated_savings": estimated_electricity_savings,
            "natural_gas_estimated_savings": estimated_gas_savings,
            "electricity_pre_params": elec_pre_params,
            "natural_gas_pre_params": gas_pre_params,
            "electricity_post_params": elec_post_params,
            "natural_gas_post_params": gas_post_params,
        }

        return results

    def _generate_fuel_consumptions(self, weather_source, weather_normal_source, periods,
            model, param_dists, param_delta_dists, noise,
            retrofit_start_date, retrofit_completion_date,
            fuel_type, consumption_unit_name, temperature_unit_name):

        pre_params = {}
        post_params = {}
        for k,v in param_dists.items():
            pre_params[k] = v.rvs()
            post_params[k] = pre_params[k] + param_delta_dists[k].rvs()

        annualized_usage_meter = AnnualizedUsageMeter(temperature_unit_name,model)
        pre_annualized_usage = annualized_usage_meter.evaluate(
                temp_sensitivity_params=model.param_dict_to_list(pre_params),
                weather_normal_source=weather_normal_source)["annualized_usage"]
        post_annualized_usage = annualized_usage_meter.evaluate(
                temp_sensitivity_params=model.param_dict_to_list(post_params),
                weather_normal_source=weather_normal_source)["annualized_usage"]
        estimated_annualized_savings = pre_annualized_usage - post_annualized_usage

        pre_generator = ConsumptionGenerator(fuel_type, consumption_unit_name,
                temperature_unit_name, model, pre_params)
        post_generator = ConsumptionGenerator(fuel_type, consumption_unit_name,
                temperature_unit_name, model, post_params)

        pre_consumptions = pre_generator.generate(weather_source, periods, noise)
        post_consumptions = post_generator.generate(weather_source, periods, noise)

        final_consumptions = []
        for pre_c, post_c, period in zip(pre_consumptions,post_consumptions,periods):
            pre_retrofit_completion = period.start < retrofit_completion_date
            post_retrofit_start = period.end > retrofit_start_date
            if not pre_retrofit_completion:
                consumption = post_c
            elif not post_retrofit_start:
                consumption = pre_c
            else:
                usage = (pre_c.to(consumption_unit_name) + post_c.to(consumption_unit_name)) / 2
                consumption = Consumption(usage,consumption_unit_name,fuel_type,period.start,period.end)

            final_consumptions.append(consumption)

        return final_consumptions, estimated_annualized_savings, pre_params, post_params

def generate_periods(start_datetime,end_datetime,period_length_mean=timedelta(days=30),period_jitter=timedelta(days=1),jitter_intensity=3):
    """Returns an array of random, variable-length DatetimePeriods for more
    realistic simulation of projects and portfolios.
    """
    assert start_datetime < end_datetime
    periods = []
    previous_datetime = start_datetime
    while True:
        next_datetime = previous_datetime + period_length_mean + (period_jitter * random.randint(-jitter_intensity,jitter_intensity))
        if next_datetime < end_datetime:
            periods.append(DatetimePeriod(previous_datetime,next_datetime))
            previous_datetime = next_datetime
        else:
            break
    return periods

