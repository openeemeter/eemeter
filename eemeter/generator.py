from consumption import Consumption
from datetime import timedelta
from eemeter.consumption import DatetimePeriod
import random
import numpy as np

class ConsumptionGenerator:
    def __init__(self, fuel_type, consumption_unit_name, temperature_unit_name,
            model, params):
        self.fuel_type = fuel_type
        self.consumption_unit_name = consumption_unit_name
        self.temperature_unit_name = temperature_unit_name

        self.model = model
        self.params = params

    def generate(self, weather_source, periods, noise=None):
        """
        `noise` is an instance of scipy.stats.rv_continuous, e.g. scipy.stats.normal()
        noise is additive and sampled independently for each period

        """
        period_daily_temps = weather_source.get_daily_temperatures(periods,self.temperature_unit_name)

        usages = self.model.compute_usage_estimates(self.params,period_daily_temps)

        consumptions = []
        for u,period in zip(usages,periods):
            if noise is not None:
                u += noise.rvs()

            c = Consumption(u, self.consumption_unit_name, self.fuel_type, period.start, period.end)
            consumptions.append(c)

        return consumptions

class ProjectGenerator:
    def __init__(self, electricity_model, gas_model,
            electricity_param_distributions, electricity_param_delta_distributions,
            gas_param_distributions, gas_param_delta_distributions,
            temperature_unit_name="degF"):
        self.electricity_model = electricity_model
        self.gas_model = gas_model
        self.elec_param_dists = electricity_param_distributions
        self.elec_param_delta_dists = electricity_param_delta_distributions
        self.gas_param_dists = gas_param_distributions
        self.gas_param_delta_dists = gas_param_delta_distributions
        self.temperature_unit_name = temperature_unit_name

    def generate(self, weather_source, electricity_periods, gas_periods,
            retrofit_start_date, retrofit_completion_date,
            electricity_noise=None,gas_noise=None):

        electricity_consumptions = self._generate_fuel_consumptions(
                weather_source, electricity_periods,
                self.electricity_model, self.elec_param_dists,
                self.elec_param_delta_dists, electricity_noise,
                retrofit_start_date, retrofit_completion_date,
                "electricity", "kWh", self.temperature_unit_name)

        gas_consumptions = self._generate_fuel_consumptions(
                weather_source, gas_periods,
                self.gas_model, self.gas_param_dists,
                self.gas_param_delta_dists, gas_noise,
                retrofit_start_date, retrofit_completion_date,
                "natural_gas", "therm", self.temperature_unit_name)

        return electricity_consumptions, gas_consumptions

    def _generate_fuel_consumptions(self, weather_source, periods,
            model, param_dists, param_delta_dists, noise,
            retrofit_start_date, retrofit_completion_date,
            fuel_type, consumption_unit_name, temperature_unit_name):

        pre_params = np.array([pd.rvs() for pd in param_dists])
        param_deltas = np.array([pd.rvs() for pd in param_delta_dists])
        post_params = pre_params + param_deltas

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

        return final_consumptions

def generate_periods(start_datetime,end_datetime,period_length_mean=timedelta(days=30),period_jitter=timedelta(days=1),jitter_intensity=3):
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

