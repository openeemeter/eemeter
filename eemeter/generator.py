from datetime import timedelta
from eemeter.consumption import ConsumptionData
from eemeter.evaluation import Period
from eemeter.project import Project

from eemeter.meter import AnnualizedUsageMeter
from eemeter.weather import GSODWeatherSource
from eemeter.weather import TMY3WeatherSource

import random
import numpy as np
import pandas as pd

from scipy.stats import poisson

class MonthlyBillingConsumptionGenerator:
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
        self.params = self.model.param_type(params)

    def generate(self, weather_source, datetimes, daily_noise_dist=None):
        """Returns a ConsumptionData instance given a particular weather
        source and a list of datetimes.

        Parameters
        ----------
        weather_source : eemeter.weather.WeatherSourceBase
            Weather source from which to draw outdoor temperature data.
        datetimes : list of datetime objects
            Periods over which to simulate consumption.
        daily_noise_dist : scipy.stats.rv_continuous, default None
            Noise to add to each day in a period. Noise is additive and sampled
            independently for each period. e.g. scipy.stats.normal().
        """
        records = [{"start": start, "end": end, "value": np.nan}
                for start, end in zip(datetimes, datetimes[1:])]
        cd = ConsumptionData(records, self.fuel_type,
                self.consumption_unit_name, record_type="arbitrary")

        periods = cd.periods()
        period_daily_temps = weather_source.daily_temperatures(periods,
                self.temperature_unit_name)

        period_average_daily_usages = self.model.transform(period_daily_temps,self.params)

        for average_daily_usage, period in zip(period_average_daily_usages,periods):
            n_days = period.timedelta.days
            if daily_noise_dist is not None:
                average_daily_usage += np.mean(daily_noise_dist.rvs(n_days))
            cd.data[period.start] = average_daily_usage * n_days

        return cd

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
        self.elec_param_dists = electricity_model.param_type(electricity_param_distributions)
        self.elec_param_delta_dists = electricity_model.param_type(electricity_param_delta_distributions)

        self.gas_model = gas_model
        self.gas_param_dists = gas_model.param_type(gas_param_distributions)
        self.gas_param_delta_dists = gas_model.param_type(gas_param_delta_distributions)

        self.temperature_unit_name = temperature_unit_name

    def generate(self, location, period_elec, period_gas,
            baseline_period, reporting_period,
            noise_elec=None, noise_gas=None):
        """Returns a simple simulated project consisting of generated
        electricity and gas consumptions and estimated savings for each.

        Parameters
        ----------
        location : eemeter.location.Location
            Location of project
        """
        early_date = None
        late_date = None

        if not period_elec.closed or not period_gas.closed:
            message = "Periods of consumption must have start and end."
            raise ValueError(message)

        all_periods = [period_elec, period_gas, baseline_period,
                reporting_period]
        for period in all_periods:
            if early_date is None:
                early_date = period.start
            if late_date is None:
                late_date = period.end
            if period.start is not None and period.start < early_date:
                early_date = period.start
            if period.end is not None and late_date < period.end:
                late_date = period.end

        weather_source = GSODWeatherSource(location.station,early_date.year,
                late_date.year)
        weather_normal_source = TMY3WeatherSource(location.station)

        cd_elec, est_savings_elec, elec_bl_params, elec_rp_params = \
                self._generate_fuel_consumptions(
                        weather_source, weather_normal_source, period_elec,
                        self.electricity_model, self.elec_param_dists,
                        self.elec_param_delta_dists, noise_elec,
                        baseline_period, reporting_period,
                        "electricity", "kWh", self.temperature_unit_name)

        cd_gas, est_savings_gas, gas_bl_params, gas_rp_params = \
                self._generate_fuel_consumptions(
                        weather_source, weather_normal_source, period_gas,
                        self.gas_model, self.gas_param_dists,
                        self.gas_param_delta_dists, noise_gas,
                        baseline_period, reporting_period,
                        "natural_gas", "therm", self.temperature_unit_name)

        project = Project(location, [cd_elec, cd_gas], baseline_period,
                reporting_period)

        results = {
            "project": project,
            "electricity_estimated_savings": est_savings_elec,
            "natural_gas_estimated_savings": est_savings_gas,
            "electricity_pre_params": elec_bl_params,
            "natural_gas_pre_params": gas_bl_params,
            "electricity_post_params": elec_rp_params,
            "natural_gas_post_params": gas_rp_params,
        }

        return results

    def _generate_fuel_consumptions(self, weather_source,
            weather_normal_source, period, model, param_dists,
            param_delta_dists, noise, baseline_period, reporting_period,
            fuel_type, consumption_unit_name, temperature_unit_name):

        baseline_params = model.param_type([param.rvs() for param in param_dists.to_list()])
        reporting_params = model.param_type([bl_param + param_delta.rvs()
            for bl_param, param_delta in zip(baseline_params.to_list(), param_delta_dists.to_list())])

        annualized_usage_meter = AnnualizedUsageMeter(temperature_unit_name,
                model)
        baseline_annualized_usage = annualized_usage_meter.evaluate_raw(
                model_params=baseline_params,
                weather_normal_source=weather_normal_source)["annualized_usage"]
        reporting_annualized_usage = annualized_usage_meter.evaluate_raw(
                model_params=reporting_params,
                weather_normal_source=weather_normal_source)["annualized_usage"]
        estimated_annualized_savings = baseline_annualized_usage - \
                reporting_annualized_usage

        baseline_generator = MonthlyBillingConsumptionGenerator(fuel_type,
                consumption_unit_name, temperature_unit_name, model,
                baseline_params)
        reporting_generator = MonthlyBillingConsumptionGenerator(fuel_type,
                consumption_unit_name, temperature_unit_name, model,
                reporting_params)

        datetimes = generate_monthly_billing_datetimes(period, dist=None)

        baseline_consumption_data = baseline_generator.generate(
                weather_source, datetimes, daily_noise_dist=noise)
        reporting_consumption_data = reporting_generator.generate(
                weather_source, datetimes, daily_noise_dist=noise)

        baseline_data = baseline_consumption_data.data
        reporting_data = reporting_consumption_data.data
        periods = baseline_consumption_data.periods()

        records = []
        for bl_data, rp_data, period in zip(baseline_data, reporting_data,
                periods):
            if period in reporting_period:
                val = rp_data
            else:
                val = bl_data
            record = {"start": period.start, "end": period.end, "value": val}
            records.append(record)

        cd = ConsumptionData(records, fuel_type, consumption_unit_name,
                record_type="arbitrary")

        return cd, estimated_annualized_savings, baseline_params, \
                reporting_params

def generate_monthly_billing_datetimes(period, dist=None):
    """Returns an array of poisson distributed datetimes falling on simulated
    monthly billing dates.

    Parameters
    ----------

    period : eemeter.evaluation.Period
        The period over which to generate datetimes. Datetimes will all fall
        within the bounds of this period, and will start on the start date of
        the period. Must be on a closed interval.
    dist : scipy.stats.rv_discrete, default None
        The distribution from which to draw samples of number of days between
        monthly bills. If :code:`None`, defaults to
        :code:`scipy.stats.poisson(365/12.)`.

    """
    # make sure period is closed
    period_delta = period.timedelta
    if period_delta is None:
        message = "Please provide a period with valid start and end date."
        raise ValueError(message)

    # The default distribution is poisson, which is a discrete distribution
    # often used to simulate rare events in nature with a particular average
    # frequency. In this case, we're using an average frequency of 12 times per
    # 365 days.
    if dist is None:
        dist = poisson(365/12.)

    periods = [period.start]
    while True:
        next_date = periods[-1] + timedelta(days=dist.rvs())
        if next_date < period.end:
            periods.append(next_date)
        else:
            break
    return periods

def generate_interval_datetimes(period, freq):
    """Returns an array of equally-spaced dates that simulate AMI periods.

    Parameters
    ----------

    period : eemeter.evaluation.Period
        The period over which to generate periods. Periods will fall within
        the bounds of this period, and will start on the start date of the
        period. Must be on a closed interval.
    freq : str
        Should be a period specification following the syntax of pandas
        offset aliases.

    """
    # make sure period is closed
    if period.start is None or period.end is None:
        message = "Please provide a period with valid start and end date."
        raise ValueError(message)

    return [d for d in pd.date_range(period.start,period.end,freq=freq)]
