import scipy.optimize as opt
import numpy as np

from datetime import timedelta
from datetime import datetime
from eemeter.consumption import DatetimePeriod

import inspect
from itertools import chain

from pprint import pprint

class MeterBase(object):
    """Base class for all Meter objects. Takes care of structural tasks such as
    input and output mapping.
    """
    def __init__(self,input_mapping={},output_mapping={},extras={},**kwargs):
        """- `input_mapping` should be a dictionary with incoming input names
             as keys whose associated values are outgoing input names.
             (e.g. ({"old_input_name":"new_input_name"})
           - `output_mapping` should be a dictionary with incoming output names
             as keys whose associated values are outgoing output names.
             (e.g. ({"old_output_name":"new_output_name"})
        """
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.extras = extras

    def evaluate(self,**kwargs):
        """Returns a dictionary of evaluated meter outputs. Arguments must be
        specified as keyword arguments. **Note:** because `**kwargs` is
        intentionally general (and therefore woefully unspecific), the function
        `meter.get_inputs()` exists to help describe the required inputs.
        """
        mapped_inputs = self._remap(kwargs,self.input_mapping)
        for k,v in self.extras.items():
            if k in mapped_inputs:
                message = "duplicate key '{}' found while adding extras to inputs"\
                        .format(k)
                raise ValueError(message)
            else:
                mapped_inputs[k] = v
        inputs = self.get_inputs()[self.__class__.__name__]["inputs"]
        for inpt in inputs:
            if inpt not in mapped_inputs:
                message = "expected argument '{}' for meter '{}'; "\
                          "got kwargs={} (with mapped_inputs={}) instead."\
                                  .format(inpt,self.__class__.__name__,
                                          sorted(kwargs.items()),sorted(mapped_inputs.items()))
                raise TypeError(message)
        result = self.evaluate_mapped_inputs(**mapped_inputs)
        mapped_outputs = self._remap(result,self.output_mapping)
        for k,v in self.extras.items():
            if k in mapped_outputs:
                message = "duplicate key '{}' found while adding extras to inputs"\
                        .format(k)
                raise ValueError(message)
            else:
                mapped_outputs[k] = v
        return mapped_outputs

    @staticmethod
    def _remap(inputs,mapping):
        """Returns a dictionary with mapped keys. `mapping` should be a
        dictionary with incoming input names as keys whose associated values
        are outgoing input names. (e.g. ({"old_key":"new_key"})
        """
        mapped_dict = {}
        for k,v in inputs.items():
            mapped_keys = mapping.get(k,k)
            if mapped_keys is not None:
                if isinstance(mapped_keys, basestring):
                    mapped_keys = [mapped_keys]
                for mapped_key in mapped_keys:
                    if mapped_key in mapped_dict:
                        message = "duplicate key '{}' found while mapping"\
                                .format(mapped_key)
                        raise ValueError(message)
                    mapped_dict[mapped_key] = v
        return mapped_dict

    def evaluate_mapped_inputs(self,**kwargs):
        """Should be the workhorse method which implements the logic of the
        meter, returning a dictionary of meter outputs.
        """
        raise NotImplementedError

    def get_inputs(self):
        """Returns a recursively structured object which shows necessary
        inputs.
        """
        inputs = inspect.getargspec(self.evaluate_mapped_inputs).args[1:]
        child_inputs = self._get_child_inputs()
        return {self.__class__.__name__:{"inputs":inputs,"child_inputs":child_inputs}}

    def _get_child_inputs(self):
        return []

class Sequence(MeterBase):
    def __init__(self,sequence,**kwargs):
        super(Sequence,self).__init__(**kwargs)
        assert all([issubclass(meter.__class__,MeterBase)
                    for meter in sequence])
        self.sequence = sequence

    def evaluate_mapped_inputs(self,**kwargs):
        """Collects and returns a series of meter object evaluation outputs in
        sequence, making the outputs of each meter available to those that
        follow.
        """
        result = {}
        for meter in self.sequence:
            args = {k:v for k,v in chain(kwargs.items(),result.items())}
            meter_result = meter.evaluate(**args)
            for k,v in meter_result.items():
                if k in result:
                    message = "unexpected repeated metric ({}) in {}. " \
                              "A different input_mapping or " \
                              "output_mapping may fix this overlap.".format(k,meter)
                    raise ValueError(message)
                result[k] = v
        return result

    def _get_child_inputs(self):
        inputs = []
        for meter in self.sequence:
            inputs.append(meter.get_inputs())
        return inputs

class Condition(MeterBase):
    def __init__(self,condition_parameter,success=None,failure=None,**kwargs):
        super(Condition,self).__init__(**kwargs)
        self.condition_parameter = condition_parameter
        self.success = success
        self.failure = failure

    def evaluate_mapped_inputs(self,**kwargs):
        """Returns evaluations for either the `success` meter or the `failure`
        meter depending on the boolean value of the meter input or output with
        the name stored in `condition_parameter`.
        """
        if kwargs[self.condition_parameter]:
            if self.success is None:
                return {}
            else:
                return self.success.evaluate(**kwargs)
        else:
            if self.failure is None:
                return {}
            else:
                return self.failure.evaluate(**kwargs)

    def _get_child_inputs(self):
        inputs = {}
        if self.success is not None:
            inputs["success"] = self.success.get_inputs()
        if self.failure is not None:
            inputs["failure"] = self.success.get_inputs()
        return inputs

class And(MeterBase):
    def __init__(self,inputs,**kwargs):
        super(And,self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_mapped_inputs(self,**kwargs):
        output = True
        for inpt in self.inputs:
            boolean = kwargs.get(inpt)
            if boolean is None:
                message = "could not find input '{}'".format(inpt)
                raise ValueError(message)
            output = output and boolean
        return {"output": output}

class Or(MeterBase):
    def __init__(self,inputs,**kwargs):
        super(Or,self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_mapped_inputs(self,**kwargs):
        output = False
        for inpt in self.inputs:
            boolean = kwargs.get(inpt)
            if boolean is None:
                message = "could not find input '{}'".format(inpt)
                raise ValueError(message)
            output = output or boolean
        return {"output": output}


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

class PrePost(MeterBase):
    def __init__(self,meter,splittable_args,**kwargs):
        super(PrePost,self).__init__(**kwargs)
        self.meter = meter
        self.splittable_args = splittable_args

    def evaluate_mapped_inputs(self,retrofit_start_date,retrofit_end_date,**kwargs):
        """Splits consuption_history into pre and post retrofit periods, then
        evaluates a meter on each subset of consumptions, appending the strings
        `"_pre"` and `"_post"`, respectively, to each key of each meter output.
        """
        pre_kwargs = {}
        post_kwargs = {}
        split_kwargs = {}
        for k,v in kwargs.items():
            if k in self.splittable_args:
                pre_kwargs[k] = v.before(retrofit_start_date)
                post_kwargs[k] = v.after(retrofit_end_date)
                split_kwargs[k + "_pre"] = pre_kwargs[k]
                split_kwargs[k + "_post"] = post_kwargs[k]
            else:
                pre_kwargs[k] = kwargs[k]
                post_kwargs[k] = kwargs[k]
        pre_results = self.meter.evaluate(**pre_kwargs)
        post_results = self.meter.evaluate(**post_kwargs)
        pre_results = {k + "_pre":v for k,v in pre_results.items()}
        post_results = {k + "_post":v for k,v in post_results.items()}
        results = {k:v for k,v in chain(pre_results.items(),
                                        post_results.items(),
                                        split_kwargs.items())}
        return results

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
            inputs = dict(kwargs.items() + {"fuel_type": fuel_type}.items())
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

class MeetsThresholds(MeterBase):
    def __init__(self,values,thresholds,operations,proportions,output_names,**kwargs):
        super(MeetsThresholds,self).__init__(**kwargs)
        self.values = values
        self.thresholds = thresholds
        self.operations = operations
        self.proportions = proportions
        self.output_names = output_names

    def evaluate_mapped_inputs(self,**kwargs):
        result = {}
        for v,t,o,p,n in zip(self.values,self.thresholds,self.operations,self.proportions,self.output_names):
            value = kwargs.get(v)
            if isinstance(t,basestring):
                threshold = kwargs.get(t)
            else:
                threshold = t
            if o == "lt":
                result[n] = (value < threshold * p)
            elif o == "gt":
                result[n] = (value > threshold * p)
            elif o == "lte":
                result[n] = (value <= threshold * p)
            elif o == "gte":
                result[n] = (value >= threshold * p)
        return result

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

class Switch(MeterBase):
    def __init__(self,target,cases,default=None,**kwargs):
        super(Switch,self).__init__(**kwargs)
        self.target = target
        self.cases = cases
        self.default = default

    def evaluate_mapped_inputs(self,**kwargs):
        item = kwargs.get(self.target)
        if item is None:
            return {}
        meter = self.cases.get(item)
        if meter is not None:
            return meter.evaluate(**kwargs)
        if self.default is not None:
            return self.default.evaluate(**kwargs)
        return {}

class Debug(MeterBase):
    def evaluate_mapped_inputs(self,**kwargs):
        """Helpful for debugging meter instances - prints out kwargs for
        inspection.
        """
        print("DEBUG")
        pprint(kwargs)
        return {}

class DummyMeter(MeterBase):
    def evaluate_mapped_inputs(self,value,**kwargs):
        """Helpful for testing meters - passes a value directly through. May
        also be helpful for hacking input/output mappings.
        """
        result = {"result": value}
        return result
