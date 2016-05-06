from .base import MeterBase

from eemeter.consumption import ConsumptionData
from datetime import datetime
from datetime import timedelta
from eemeter.evaluation import Period
import pytz

from itertools import chain
import numpy as np
import pandas as pd

class TemperatureSensitivityParameterOptimizationMeter(MeterBase):
    """Optimizes temperature senstivity parameter choices.

    Parameters
    ----------
    temperature_unit_str : str
        Unit of temperature, usually "degC" or "degF".
    model : eemeter.model.TemperatureSensitivityModel
        Model of energy usage for which to optimize parameter choices.
    """

    def __init__(self, temperature_unit_str, model, **kwargs):
        super(TemperatureSensitivityParameterOptimizationMeter,
                self).__init__(**kwargs)
        self.temperature_unit_str = temperature_unit_str
        self.model = model

    def evaluate_raw(self, consumption_data, weather_source,
            energy_unit_str, **kwargs):
        """Run optimization of temperature sensitivity parameters given a
        observed consumption data, and observed temperature data.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Consumption history to use as basis of model.
        weather_source : eemeter.weather.WeatherSourceBase
            Weather data source containing data covering at least the duration
            of the consumption data.
        energy_unit_str : str
            Unit of energy, usually "kWh" or "therms".

        Returns
        -------
        out : dict
            - "temp_sensitivity_params": an array of optimal parameters
            - "average_daily_usages": an array of actual average daily usages
            - "estimated_average_daily_usages": an array of estimated usages
              as given by the model.
            - "n_days": an array of the number of days in each consumption
              period (weights)
        """
        average_daily_usages, n_days = \
                consumption_data.average_daily_consumptions()
        periods = consumption_data.periods()
        observed_daily_temps = weather_source.daily_temperatures(periods,
                self.temperature_unit_str)

        params = self.model.fit(observed_daily_temps, average_daily_usages, weights=n_days)

        estimated_daily_usages = self.model.transform(observed_daily_temps, params)

        return {"temp_sensitivity_params": params,
                "average_daily_usages": average_daily_usages,
                "estimated_average_daily_usages": estimated_daily_usages,
                "n_days": n_days}

class AnnualizedUsageMeter(MeterBase):
    """Weather normalizes modeled usage for an annualized estimate of
    consumption.

    Parameters
    ----------
    temperature_unit_str : str
        Unit of temperature, usually "degC" or "degF".
    model : eemeter.model.AverageDailyTemperatureSensitivityModel
        Model of energy usage
    """

    def __init__(self, temperature_unit_str, model, **kwargs):
        super(AnnualizedUsageMeter,self).__init__(**kwargs)
        self.temperature_unit_str = temperature_unit_str
        self.model = model

    def evaluate_raw(self, model_params,
            weather_normal_source, **kwargs):
        """Evaluates the annualized usage metric given a particular set of
        model parameters and a weather normal source.

        Parameters
        ----------
        model_params : object
            Parameters in a format recognized by the model
            `compute_usage_estimates` method.
        weather_normal_source : eemeter.weather.WeatherBase and eemeter.weather.WeatherNormalMixin
            Weather normal data source. Should be from a location (station) as
            geographically and climatically similar to project as possible.

        Returns
        -------
        out : dict
            - "annualized_usage": annualized usage given temperature
              sensitivity parameters and weather normals.
        """
        daily_temps = weather_normal_source.annual_daily_temperatures(
                self.temperature_unit_str)
        average_daily_usage_estimate = self.model.transform(daily_temps, model_params)

        annualized_usage = average_daily_usage_estimate * 365
        return {"annualized_usage": annualized_usage}

class GrossSavingsMeter(MeterBase):
    """Calculates savings due to an efficiency retrofit or ECM for a particular
    fuel type using a conterfactual usage estimate and actual usage.

    Parameters
    ----------
    model : eemeter.model.AverageDailyTemperatureSensitivityModel
        Model of energy usage
    temperature_unit_str : str
        Unit of temperature, usually "degC" or "degF".
    """

    def __init__(self, model, temperature_unit_str,
            **kwargs):
        super(GrossSavingsMeter, self).__init__(**kwargs)
        self.model = model
        self.temperature_unit_str = temperature_unit_str

    def evaluate_raw(self, model_params_baseline, consumption_data_reporting,
            weather_source, energy_unit_str, **kwargs):
        """Evaluates the gross savings metric.

        Parameters
        ----------
        model_params_baseline : object
            Parameters in a format recognized by the model
            `compute_usage_estimates` method.
        consumption_data_reporting : eemeter.consumption.ConsumptionData
            Consumption periods in reporting period over which gross savings
            will be estimated.
        weather_source : eemeter.weather.WeatherSourceBase
            Weather data source containing data covering at least the duration
            of the consumption data.
        energy_unit_str : str
            Unit of energy, usually "kWh" or "therms".

        Returns
        -------
        out : dict
            - "gross_savings": Total cumulative savings over reporting period.
        """
        consumption_periods = consumption_data_reporting.periods()
        consumption_reporting = consumption_data_reporting.to(energy_unit_str)[:-1]
        observed_daily_temps = weather_source.daily_temperatures(
                consumption_periods, self.temperature_unit_str)
        consumption_estimates_baseline = self.model.transform( observed_daily_temps, model_params_baseline)
        consumption_estimates_baseline *= np.array([p.timedelta.days for p in consumption_periods])
        gross_savings = np.nansum(consumption_estimates_baseline -
                consumption_reporting)
        return {"gross_savings": gross_savings}

class AnnualizedGrossSavingsMeter(MeterBase):
    """Annualized gross savings accumulated since the completion of a retrofit
    calculated by multiplying an annualized savings estimate by the number
    of years since retrofit completion.

    Parameters
    ----------
    model : eemeter.model.AverageDailyTemperatureSensitivityModel
        Model of energy usage
    temperature_unit_str : str
        Unit of temperature, usually "degC" or "degF".
    """

    def __init__(self, model, temperature_unit_str, **kwargs):
        super(AnnualizedGrossSavingsMeter,self).__init__(**kwargs)
        self.model = model
        self.temperature_unit_str = temperature_unit_str

    def evaluate_raw(self, model_params_baseline,
            model_params_reporting, consumption_data_reporting,
            weather_normal_source, **kwargs):
        """Evaluates the annualized gross savings metric.

        Parameters
        ----------
        model_params_baseline : object
            Parameters for baseline period in a format recognized by the
            model `compute_usage_estimates` method.
        model_params_reporting : object
            Parameters for reporting period in a format recognized by the
            model `compute_usage_estimates` method.
        consumption_data_reporting : eemeter.consumption.ConsumptionData
            Consumption data over which annualized gross savings will be
            estimated. (Note: only used for finding appropriate number of days
            multiplier).
        weather_normal_source : eemeter.weather.WeatherBase and eemeter.weather.WeatherNormalMixin
            Weather normal data source. Should be from a location (station) as
            geographically and climatically similar to project as possible.

        Returns
        -------
        out : dict
            - "annualized_gross_savings": Annualized savings over reporting
              period.
        """

        meter = AnnualizedUsageMeter(self.temperature_unit_str, self.model)
        annualized_consumption_baseline = meter.evaluate_raw(
                model_params=model_params_baseline,
                weather_normal_source=weather_normal_source)["annualized_usage"]
        annualized_consumption_reporting = meter.evaluate_raw(
                model_params=model_params_reporting,
                weather_normal_source=weather_normal_source)["annualized_usage"]
        annualized_avoided_consumption = annualized_consumption_baseline - \
                annualized_consumption_reporting
        n_years = consumption_data_reporting.total_days()/365.
        annualized_gross_savings = n_years * annualized_avoided_consumption
        return {"annualized_gross_savings": annualized_gross_savings}

class TimeSpanMeter(MeterBase):
    """Meters the time span (in days) of a ConsumptionData instance.
    """
    def __init__(self, **kwargs):
        super(TimeSpanMeter, self).__init__(**kwargs)

    def evaluate_raw(self, consumption_data, **kwargs):
        """Evaluates a ConsumptionData instance to determine the number of
        unique days covered by consumption periods

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Target of time span analysis

        Returns
        -------
        out : dict
            - "time_span": the number of days covered by the consumption data.
        """
        return { "time_span": consumption_data.total_days() }

class TotalHDDMeter(MeterBase):
    """Sums the total heating degree days observed over the course of a
    ConsumptionData instance

    Parameters
    ----------
    base : int or float
        The heating degree day base.
    temperature_unit_str : {"degF", "degC"}
        A string denoting the temperature unit to be used.
    """
    def __init__(self,base,temperature_unit_str,**kwargs):
        super(TotalHDDMeter,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_raw(self, consumption_data, weather_source, **kwargs):
        """Sums the total observed HDD over a consumption history.

        Parameters
        ----------
        consumption_data : eemeter.meter.ConsumptionData
            The consumption data over which to sum heating degree days
        weather_source : eemeter.weather.WeatherSourceBase
            A weather data source from a location as geographically and
            climatically close as possible to the target project.

        Returns
        -------
        out : dict
            - "total_hdd": total heating degree days observed during the time
              span covered by the consumption_data instance.
        """
        consumption_periods = consumption_data.periods()
        hdd = weather_source.hdd(consumption_periods,
                self.temperature_unit_str, self.base)
        return { "total_hdd": sum(hdd) }

class TotalCDDMeter(MeterBase):
    """Sums the total cooling degree days observed over the course of a
    ConsumptionData instance

    Parameters
    ----------
    base : int or float
        The cooling degree day base.
    temperature_unit_str : {"degF", "degC"}
        A string denoting the temperature unit to be used.
    """
    def __init__(self, base, temperature_unit_str, **kwargs):
        super(TotalCDDMeter,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_raw(self, consumption_data, weather_source,
            **kwargs):
        """Sums the total observed CDD over a consumption history.

        Parameters
        ----------
        consumption_data : eemeter.meter.ConsumptionData
            The consumption data over which to sum cooling degree days
        weather_source : eemeter.weather.WeatherSourceBase
            A weather data source from a location as geographically and
            climatically close as possible to the target project.

        Returns
        -------
        out : dict
            - "total_cdd": total cooling degree days observed during the time
              span covered by the consumption_data instance.
        """
        consumption_periods = consumption_data.periods()
        cdd = weather_source.cdd(consumption_periods,
                self.temperature_unit_str, self.base)
        return { "total_cdd": sum(cdd) }


class NormalAnnualHDD(MeterBase):
    """Sums the total heating degree days observed in a normal year.

    Parameters
    ----------
    base : int or float
        The heating degree day base.
    temperature_unit_str : {"degF", "degC"}
        A string denoting the temperature unit to be used.
    """
    def __init__(self, base, temperature_unit_str, **kwargs):
        super(NormalAnnualHDD,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_raw(self, weather_normal_source, **kwargs):
        """Sums the total observed HDD in a normal year

        Parameters
        ----------
        weather_normal_source : eemeter.weather.WeatherSourceBase and eemeter.weather.WeatherNormalMixin
            A weather data source from a location as geographically and
            climatically close as possible to the target project.

        Returns
        -------
        out : dict
            - "normal_annual_hdd": the total heating degree days observed
              during a typical meteorological year
        """
        # year of this annual period will be ignored
        annual_period = Period(datetime(2013,1,1), datetime(2014,1,1))
        hdd = weather_normal_source.hdd(annual_period,
                self.temperature_unit_str, self.base)
        return { "normal_annual_hdd": hdd }

class NormalAnnualCDD(MeterBase):
    """Sums the total cooling degree days observed in a normal year.

    Parameters
    ----------
    base : int or float
        The heating degree day base.
    temperature_unit_str : {"degF", "degC"}
        A string denoting the temperature unit to be used.
    """
    def __init__(self, base, temperature_unit_str, **kwargs):
        super(NormalAnnualCDD,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_raw(self, weather_normal_source, **kwargs):
        """Sums the total observed CDD in a normal year

        Parameters
        ----------
        weather_normal_source : eemeter.weather.WeatherSourceBase and eemeter.weather.WeatherNormalMixin
            A weather data source from a location as geographically and
            climatically close as possible to the target project.

        Returns
        -------
        out : dict
            - "normal_annual_hdd": the total cooling degree days observed
              during a typical meteorological year
        """

        annual_period = Period(datetime(2013,1,1), datetime(2014,1,1))
        cdd = weather_normal_source.cdd(annual_period,
                self.temperature_unit_str, self.base)
        return { "normal_annual_cdd": cdd }

class NPeriodsMeetingHDDPerDayThreshold(MeterBase):
    """Counts the number of periods meeting a particular heating degree day
    threshold.

    Parameters
    ----------
    base : int or float
        The heating degree day base.
    temperature_unit_str : {"degF", "degC"}
        A string denoting the temperature unit to be used.
    operation : {"<", "<=", ">", ">="}
        A string representing the type of inequality test. (I.e. Is the
        threshold an upper or lower bound? Is the endpoint included?)
    proportion : float, optional
        A proportion multiplier for the number of hdd; defualt is 1.
        E.g. period_hdd <= proportion * hdd:
    """
    def __init__(self, base, temperature_unit_str, operation, proportion=1,
            **kwargs):
        super(NPeriodsMeetingHDDPerDayThreshold,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str
        self.operation = operation
        self.proportion = proportion

    def evaluate_raw(self, consumption_data, hdd, weather_source, **kwargs):
        """Evaluates the number of periods meeting a consumption history limit
        according to data from a particular weather source.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Consumption data over which to count periods
        hdd : int or float
            The target number of HDD.
        weather_source : eemeter.weather.WeatherSourceBase
            A weather data source from a location as geographically and
            climatically close as possible to the target project.

        Returns
        -------
        out : dict
            - "n_periods": the number of periods meeting the threshold.
        """
        n_periods = 0
        periods = consumption_data.periods()
        hdds = weather_source.hdd(periods, self.temperature_unit_str,
                self.base, per_day=True)
        for period_hdd in hdds:
            if self.operation == "<":
                if period_hdd < self.proportion * hdd:
                    n_periods += 1
            elif self.operation == "<=":
                if period_hdd <= self.proportion * hdd:
                    n_periods += 1
            elif self.operation == ">":
                if period_hdd > self.proportion * hdd:
                    n_periods += 1
            elif self.operation == ">=":
                if period_hdd >= self.proportion * hdd:
                    n_periods += 1
        return {"n_periods": n_periods}

class NPeriodsMeetingCDDPerDayThreshold(MeterBase):
    """Counts the number of periods meeting a particular cooling degree day
    threshold.

    Parameters
    ----------
    base : int or float
        The cooling degree day base.
    temperature_unit_str : {"degF", "degC"}
        A string denoting the temperature unit to be used.
    operation : {"<", "<=", ">", ">="}
        A string representing the type of inequality test. (I.e. Is the
        threshold an upper or lower bound? Is the endpoint included?)
    proportion : float, optional
        A proportion multiplier for the number of cdd; defualt is 1.
        E.g. period_cdd <= proportion * cdd:
    """
    def __init__(self, base, temperature_unit_str, operation, proportion=1,
            **kwargs):
        super(NPeriodsMeetingCDDPerDayThreshold,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str
        self.operation = operation
        self.proportion = proportion

    def evaluate_raw(self, consumption_data, cdd, weather_source,
            **kwargs):
        """Evaluates the number of periods meeting a consumption history limit
        according to data from a particular weather source.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Consumption data over which to count periods.
        cdd : int or float
            The target number of CDD.
        weather_source : eemeter.weather.WeatherSourceBase
            A weather data source from a location as geographically and
            climatically close as possible to the target project.

        Returns
        -------
        out : dict
            - "n_periods": the number of periods meeting the threshold.
        """
        n_periods = 0
        periods = consumption_data.periods()
        cdds = weather_source.cdd(periods, self.temperature_unit_str,
                self.base, per_day=True)
        for period_cdd in cdds:
            if self.operation == "<":
                if period_cdd < self.proportion * cdd:
                    n_periods += 1
            elif self.operation == "<=":
                if period_cdd <= self.proportion * cdd:
                    n_periods += 1
            elif self.operation == ">":
                if period_cdd > self.proportion * cdd:
                    n_periods += 1
            elif self.operation == ">=":
                if period_cdd >= self.proportion * cdd:
                    n_periods += 1
        return {"n_periods": n_periods}

class RecentReadingMeter(MeterBase):
    """ Finds the number of days since the most recent reading.

    Parameters
    ----------
    n_days : int
        The target number of days since the most recent reading.
    """
    def __init__(self, **kwargs):
        super(RecentReadingMeter, self).__init__(**kwargs)

    def evaluate_raw(self, consumption_data, **kwargs):
        """Evaluates the number of days since the last non-estimated, non-null
        reading against the threshold.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Consumption data in which to find a most recent period
        since_date : datetime.datetime, optional
            The date to count from; defaults to datetime.now(pytz.utc).

        Returns
        -------
        out : dict
            - "n_days": The number of days, counted from the last date in the
              consumption_data object, since a valid (non-null, not estimated)
              meter reading.
        """
        if consumption_data.data.shape[0] > 0:
            reverse_data = consumption_data.data[::-1]
            for i,val in reverse_data.iteritems():
                if not pd.isnull(val) and not consumption_data.estimated[i]:
                    n_days = (reverse_data.index[0] - i).days
                    return {"n_days": n_days}
        return {"n_days": np.inf }

class AverageDailyUsage(MeterBase):
    """Computes average daily usage given consumption.
    """

    def evaluate_raw(self, consumption_data, energy_unit_str,
            **kwargs):
        """Compute the average daily usage for each consumption of
        a particular fuel type.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Consumption data to draw from.
        energy_unit_str : str
            Unit of energy, usually "kWh" or "therms".

        Returns
        -------
        out : dict
            - "average_daily_usages": an array of average usage values of the
              same length as the consumption_data instance.
        """
        average_daily_consumptions, _ = \
                consumption_data.average_daily_consumptions()
        return {"average_daily_usages": average_daily_consumptions}

class EstimatedAverageDailyUsage(MeterBase):
    """Computes estmiated average daily usage given consumption, a model, and
    a weather source.

    Parameters
    ----------
    temperature_unit_str : str
        Unit of temperature, usually "degC" or "degF".
    model : eemeter.model.AverageDailyTemperatureSensitivityModel
        Model of energy usage for which to optimize parameter choices.
    """

    def __init__(self, temperature_unit_str, model, **kwargs):
        super(EstimatedAverageDailyUsage,self).__init__(**kwargs)
        self.temperature_unit_str = temperature_unit_str
        self.model = model

    def evaluate_raw(self, consumption_data, weather_source,
            temp_sensitivity_params, **kwargs):
        """Compute the average daily usage for each consumption of
        a particular fuel type.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Consumption data to draw from.
        weather_source : eemeter.weather.WeatherSourceBase
            Weather data source containing data covering at least the duration
            of the consumption data.
        temp_sensitivity_params : array_like
            Parameters to use in the estimation.

        Returns
        -------
        out : dict
            - "estimated_average_daily_usages": an array of average usage
              values.
            - "n_days": the number of days in each consumption period.
        """
        periods = consumption_data.periods()
        observed_daily_temps = weather_source.daily_temperatures(periods,
                self.temperature_unit_str)
        n_days = np.array([len(temps) for temps in observed_daily_temps])
        estimated_average_daily_usages = \
                self.model.transform( observed_daily_temps, temp_sensitivity_params)
        return {"estimated_average_daily_usages": estimated_average_daily_usages,
                "n_days": n_days}

class ConsumptionDataAttributes(MeterBase):
    """ Outputs the attributes of the ConsumptionData object passed in.
    """

    def evaluate_raw(self, consumption_data, **kwargs):
        """ Exports the attributes of the consumption_data object.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Consumption data for which to return fuel type.

        Returns
        -------
        out : dict
            - "fuel_type": string describing the fuel type of the consumption
              data.
            - "unit_name": string decsribing the energy units of the
              consumption data.
            - "freq": string decsribing the frequency of the intervals (if
              applicable).
            - "freq_timedelta": timedelta decsribing the frequency of the
              intervals (if applicable).
            - "pulse_value": the pulse value at each consumption timestamp (if
              applicable.
            - "name": the name of the consumption data.
        """
        attributes = {
            "fuel_type": consumption_data.fuel_type,
            "unit_name": consumption_data.unit_name,
            "freq": consumption_data.freq,
            "freq_timedelta": consumption_data.freq_timedelta,
            "pulse_value": consumption_data.pulse_value,
            "name": consumption_data.name,
        }
        return attributes

class ProjectAttributes(MeterBase):
    """ Outputs the attributes of the Project object passed in.
    """

    def evaluate_raw(self, project, **kwargs):
        """ Finds the fuel type of the data.

        Parameters
        ----------
        project : eemeter.consumption.ConsumptionData
            Consumption data for which to return fuel type.

        Returns
        -------
        out : dict
            - "location": eemeter.location.Location object
            - "consumption": list of eemeter.consumption.ConsumptionData objects
            - "baseline_period": eemeter.evaluation.Period
            - "reporting_period": eemeter.evaluation.Period
            - "other_periods": list of eemeter.evaluation.Period objects
            - "weather_source": eemeter.weather.GSODWeatherSource
            - "weather_normal_source": eemeter.weather.TMY3WeatherSource
        """
        attributes = {
            "location": project.location,
            "consumption": project.consumption,
            "baseline_period": project.baseline_period,
            "reporting_period": project.reporting_period,
            "other_periods": project.other_periods,
            "weather_source": project.weather_source,
            "weather_normal_source": project.weather_normal_source,
        }
        return attributes

class ProjectConsumptionDataBaselineReporting(MeterBase):
    """ Splits project consumption data by period and fuel.
    """

    def evaluate_raw(self, project, **kwargs):
        """ Creates a list of tagged ConsumptionData objects for each of this
        project's fuel types in the baseline period and the reporting period.

        Parameters
        ----------
        project : eemeter.project.Project
            Project instance from which to get consumption data.

        Returns
        -------
        out : dict
            - "consumption": list of tagged ConsumptionData instances.
        """
        consumption = []

        for c in project.consumption:
            baseline_consumption_data = \
                    c.filter_by_period(project.baseline_period)
            baseline_data = {
                "value": baseline_consumption_data,
                "tags": [c.fuel_type, "baseline"]
            }
            reporting_consumption_data = \
                    c.filter_by_period(project.reporting_period)
            reporting_data = {
                "value": reporting_consumption_data,
                "tags": [c.fuel_type, "reporting"]
            }
            consumption.append(baseline_data)
            consumption.append(reporting_data)

        return { "consumption": consumption }

class ProjectFuelTypes(MeterBase):
    """
    Forms an iterator over all fuel_types within the project.
    """

    def evaluate_raw(self, project, **kwargs):
        """ Creates a list of tagged ConsumptionData objects for each of this
        project's fuel types in the baseline period and the reporting period.

        Parameters
        ----------
        project : eemeter.project.Project
            Project instance from which to get consumption data.

        Returns
        -------
        out : dict
            - "fuel_types": list of tagged strings
        """
        fuel_types = []
        for c in project.consumption:
            fuel_type = { "value": c.fuel_type, "tags": [ c.fuel_type ] }
            fuel_types.append(fuel_type)

        return { "fuel_types": fuel_types }


class DownsampleConsumption(MeterBase):
    """
    Downsamples Consumption data to specified frequency (if specified frequency
    is lower than consumption data frequency; otherwise returns a copy
    of itself.

    Parameters
    ----------
    freq : str
        Frequency to downsample to. Use a pandas offset alias (e.g. 'D').
    """

    def __init__(self, freq, **kwargs):
        self.freq = freq
        super(DownsampleConsumption, self).__init__(**kwargs)

    def evaluate_raw(self, consumption_data, **kwargs):
        """ Downsamples given consumption data object.

        Parameters
        ----------
        consumption_data : eemeter.meter.ConsumptionData
            The consumption data to downsample.

        Returns
        -------
        out : eemeter.meter.ConsumptionData
            Downsampled consumption data.
        """

        consumption_downsampled = consumption_data.downsample(self.freq)

        return { "consumption_downsampled": consumption_downsampled }
