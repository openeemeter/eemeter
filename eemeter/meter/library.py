from .base import MeterBase

from datetime import datetime
from datetime import timedelta
from eemeter.consumption import DatetimePeriod

from itertools import chain
import numpy as np

class TemperatureSensitivityParameterOptimizationMeter(MeterBase):
    def __init__(self,fuel_unit_str,fuel_type,temperature_unit_str,model,**kwargs):
        super(TemperatureSensitivityParameterOptimizationMeter,self).__init__(**kwargs)
        self.fuel_unit_str = fuel_unit_str
        self.fuel_type = fuel_type
        self.temperature_unit_str = temperature_unit_str
        self.model = model

    def evaluate_mapped_inputs(self,consumption_history,weather_source,**kwargs):
        """Returns optimal parameters of a parameter optimization given a
        particular model, observed consumption data, and observed temperature
        data. Output dictionary is `{"temp_sensitivity_params": params}`.
        """
        consumptions = consumption_history.get(self.fuel_type)
        average_daily_usages = [c.average_daily_usage(self.fuel_unit_str) for c in consumptions]
        observed_daily_temps = weather_source.get_daily_temperatures(consumptions,self.temperature_unit_str)
        weights = [c.timedelta.days for c in consumptions]
        params = self.model.parameter_optimization(average_daily_usages,observed_daily_temps, weights)

        n_daily_temps = np.array([len(temps) for temps in observed_daily_temps])
        estimated_daily_usages = self.model.compute_usage_estimates(params,observed_daily_temps)/n_daily_temps
        sqrtn = np.sqrt(len(estimated_daily_usages))

        # use nansum to ignore consumptions with missing usages
        daily_standard_error = np.nansum(np.abs(estimated_daily_usages - average_daily_usages))/sqrtn

        return {"temp_sensitivity_params": params, "daily_standard_error":daily_standard_error}

class AnnualizedUsageMeter(MeterBase):
    def __init__(self,temperature_unit_str,model,**kwargs):
        super(AnnualizedUsageMeter,self).__init__(**kwargs)
        self.temperature_unit_str = temperature_unit_str
        self.model = model

    def evaluate_mapped_inputs(self,temp_sensitivity_params,weather_normal_source,**kwargs):
        """Returns annualized usage given temperature sensitivity parameters
        and weather normals. Output dictionary is `{"annualized_usage": annualized_usage}`.
        """
        daily_temps = weather_normal_source.annual_daily_temperatures(self.temperature_unit_str)
        usage_estimates = self.model.compute_usage_estimates(temp_sensitivity_params,daily_temps)
        annualized_usage = np.nansum(usage_estimates)
        return {"annualized_usage": annualized_usage}

class GrossSavingsMeter(MeterBase):
    def __init__(self,model,fuel_unit_str,fuel_type,temperature_unit_str,**kwargs):
        super(GrossSavingsMeter,self).__init__(**kwargs)
        self.model = model
        self.fuel_type = fuel_type
        self.fuel_unit_str = fuel_unit_str
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,temp_sensitivity_params_pre,consumption_history_post,weather_source,**kwargs):
        """Returns gross savings accumulated since the completion of a retrofit
        by estimating counterfactual usage. Output dictionary is
        `{"gross_savings": gross_savings}`.
        """
        consumptions_post = consumption_history_post.get(self.fuel_type)
        observed_daily_temps = weather_source.get_daily_temperatures(consumptions_post,self.temperature_unit_str)
        usages_post = np.array([c.to(self.fuel_unit_str) for c in consumptions_post])
        usage_estimates_pre = self.model.compute_usage_estimates(temp_sensitivity_params_pre,observed_daily_temps)
        return {"gross_savings": np.nansum(usage_estimates_pre - usages_post)}

class AnnualizedGrossSavingsMeter(MeterBase):
    def __init__(self,model,fuel_type,temperature_unit_str,**kwargs):
        super(AnnualizedGrossSavingsMeter,self).__init__(**kwargs)
        self.model = model
        self.fuel_type = fuel_type
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,temp_sensitivity_params_pre,temp_sensitivity_params_post,consumption_history_post,weather_normal_source,**kwargs):
        """Returns annualized gross savings accumulated since the completion of
        a retrofit by multiplying an annualized savings estimate by the number
        of years since retrofit completion. Output dictionary is
        `{"gross_savings": gross_savings}`.
        """
        meter = AnnualizedUsageMeter(self.temperature_unit_str,self.model)
        annualized_usage_pre = meter.evaluate(temp_sensitivity_params=temp_sensitivity_params_pre,
                                              weather_normal_source=weather_normal_source)["annualized_usage"]
        annualized_usage_post = meter.evaluate(temp_sensitivity_params=temp_sensitivity_params_post,
                                               weather_normal_source=weather_normal_source)["annualized_usage"]
        annualized_usage_savings = annualized_usage_pre - annualized_usage_post
        consumptions_post = consumption_history_post.get(self.fuel_type)
        n_years = np.sum([c.timedelta.days for c in consumptions_post])/365.
        annualized_gross_savings = n_years * annualized_usage_savings
        return {"annualized_gross_savings": annualized_gross_savings}

class FuelTypePresenceMeter(MeterBase):
    def __init__(self,fuel_types,**kwargs):
        super(FuelTypePresenceMeter,self).__init__(**kwargs)
        self.fuel_types = fuel_types

    def evaluate_mapped_inputs(self,consumption_history,**kwargs):
        """Checks for fuel_type presence in a given consumption_history and
        returns a dictionary of booleans keyed by `"[fuel_type]_presence"`
        (e.g. `fuel_types = ["electricity"]` => `{'electricity_presence': False}`
        """
        results = {}
        for fuel_type in self.fuel_types:
            consumptions = consumption_history.get(fuel_type)
            results[fuel_type + "_presence"] = (consumptions != [])
        return results

class ForEachFuelType(MeterBase):
    def __init__(self,fuel_types,meter,**kwargs):
        super(ForEachFuelType,self).__init__(**kwargs)
        self.fuel_types = fuel_types
        self.meter = meter

    def evaluate_mapped_inputs(self,**kwargs):
        """Checks for fuel_type presence in a given consumption_history and
        returns a dictionary of booleans keyed by `"[fuel_type]_presence"`
        (e.g. `fuel_types = ["electricity"]` => `{'electricity_presence': False}`
        """
        results = {}
        for fuel_type in self.fuel_types:
            inputs = dict(chain(kwargs.items(),{"fuel_type": fuel_type}.items()))
            result = self.meter.evaluate(**inputs)
            for k,v in result.items():
                results[ "{}_{}".format(k,fuel_type)] = v
        return results

class TimeSpanMeter(MeterBase):
    def __init__(self,**kwargs):
        super(TimeSpanMeter,self).__init__(**kwargs)

    def evaluate_mapped_inputs(self,consumption_history,fuel_type,**kwargs):
        consumptions = consumption_history.get(fuel_type)
        dates = set()
        for c in consumptions:
            for days in range((c.end - c.start).days):
                dat = c.start + timedelta(days=days)
                dates.add(dat)
        return { "time_span": len(dates) }

class TotalHDDMeter(MeterBase):
    def __init__(self,base,temperature_unit_str,**kwargs):
        super(TotalHDDMeter,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,consumption_history,fuel_type,weather_source,**kwargs):
        consumptions = consumption_history.get(fuel_type)
        hdd = weather_source.get_hdd(consumptions,self.temperature_unit_str,self.base)
        return { "total_hdd": sum(hdd) }

class TotalCDDMeter(MeterBase):
    def __init__(self,base,temperature_unit_str,**kwargs):
        super(TotalCDDMeter,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,consumption_history,fuel_type,weather_source,**kwargs):
        consumptions = consumption_history.get(fuel_type)
        cdd = weather_source.get_cdd(consumptions,self.temperature_unit_str,self.base)
        return { "total_cdd": sum(cdd) }


class NormalAnnualHDD(MeterBase):
    def __init__(self,base,temperature_unit_str,**kwargs):
        super(NormalAnnualHDD,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,weather_normal_source,**kwargs):
        periods = []
        for days in range(365):
            start = datetime(2013,1,1) + timedelta(days=days)
            end = datetime(2013,1,1) + timedelta(days=days + 1)
            periods.append(DatetimePeriod(start,end))
        hdd = weather_normal_source.get_hdd(periods,self.temperature_unit_str,self.base)
        return { "normal_annual_hdd": sum(hdd) }

class NormalAnnualCDD(MeterBase):
    def __init__(self,base,temperature_unit_str,**kwargs):
        super(NormalAnnualCDD,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,weather_normal_source,**kwargs):
        periods = []
        for days in range(365):
            start = datetime(2013,1,1) + timedelta(days=days)
            end = datetime(2013,1,1) + timedelta(days=days + 1)
            periods.append(DatetimePeriod(start,end))
        cdd = weather_normal_source.get_cdd(periods,self.temperature_unit_str,self.base)
        return { "normal_annual_cdd": sum(cdd) }

class NPeriodsMeetingHDDPerDayThreshold(MeterBase):
    def __init__(self,base,temperature_unit_str,operation,proportion=1,**kwargs):
        super(NPeriodsMeetingHDDPerDayThreshold,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str
        self.operation = operation
        self.proportion = proportion

    def evaluate_mapped_inputs(self,consumption_history,fuel_type,hdd,weather_source,**kwargs):
        n_periods = 0
        consumptions = consumption_history.get(fuel_type)
        hdds = weather_source.get_hdd_per_day(consumptions,self.temperature_unit_str,self.base)
        for period_hdd in hdds:
            if self.operation == "lt":
                if period_hdd < self.proportion * hdd:
                    n_periods += 1
            elif self.operation == "lte":
                if period_hdd <= self.proportion * hdd:
                    n_periods += 1
            elif self.operation == "gt":
                if period_hdd > self.proportion * hdd:
                    n_periods += 1
            elif self.operation == "gte":
                if period_hdd >= self.proportion * hdd:
                    n_periods += 1
        return {"n_periods": n_periods}

class NPeriodsMeetingCDDPerDayThreshold(MeterBase):
    def __init__(self,base,temperature_unit_str,operation,proportion=1,**kwargs):
        super(NPeriodsMeetingCDDPerDayThreshold,self).__init__(**kwargs)
        self.base = base
        self.temperature_unit_str = temperature_unit_str
        self.operation = operation
        self.proportion = proportion

    def evaluate_mapped_inputs(self,consumption_history,fuel_type,cdd,weather_source,**kwargs):
        n_periods = 0
        consumptions = consumption_history.get(fuel_type)
        cdds = weather_source.get_cdd_per_day(consumptions,self.temperature_unit_str,self.base)
        for period_cdd in cdds:
            if self.operation == "lt":
                if period_cdd < self.proportion * cdd:
                    n_periods += 1
            elif self.operation == "lte":
                if period_cdd <= self.proportion * cdd:
                    n_periods += 1
            elif self.operation == "gt":
                if period_cdd > self.proportion * cdd:
                    n_periods += 1
            elif self.operation == "gte":
                if period_cdd >= self.proportion * cdd:
                    n_periods += 1
        return {"n_periods": n_periods}

class RecentReadingMeter(MeterBase):
    def __init__(self,n_days,since_date=datetime.now(),**kwargs):
        super(RecentReadingMeter,self).__init__(**kwargs)
        self.dt_target = since_date - timedelta(days=n_days)

    def evaluate_mapped_inputs(self,consumption_history,fuel_type,**kwargs):
        recent_reading = False
        for consumption in consumption_history.get(fuel_type):
            if consumption.end > self.dt_target:
                recent_reading = True
                break
        return {"recent_reading": recent_reading}

class CVRMSE(MeterBase):
    def __init__(self,model,fuel_unit_str,**kwargs):
        super(CVRMSE,self).__init__(**kwargs)
        self.model = model
        self.fuel_unit_str = fuel_unit_str

    def evaluate_mapped_inputs(self,consumption_history,weather_source,fuel_type,**kwargs):
        consumptions = consumption_history.get(fuel_type)
        weights = np.array([c.timedelta.days for c in consumptions])
        average_daily_usages = np.array([c.to(self.fuel_unit_str) for c in consumptions]) / weights
        observed_daily_temps = weather_source.get_daily_temperatures(consumptions,"degF")
        params = self.model.parameter_optimization(average_daily_usages,observed_daily_temps,weights)
        estimated_daily_usages = self.model.compute_usage_estimates(params,observed_daily_temps) / weights
        y = average_daily_usages
        y_hat = estimated_daily_usages
        y_bar = np.mean(average_daily_usages)
        n = len(consumptions)
        p = len(params)
        cvrmse = 100 * (np.sum((y - y_hat)**2) / (n - p) )**.5 / y_bar
        return {"CVRMSE": cvrmse}
