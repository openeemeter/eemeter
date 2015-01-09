import scipy.optimize as opt
import numpy as np

from itertools import chain

class MeterBase(object):
    def __init__(self,**kwargs):
        if "input_mapping" in kwargs:
            self.input_mapping = kwargs["input_mapping"]
        else:
            self.input_mapping = {}
        if "output_mapping" in kwargs:
            self.output_mapping = kwargs["output_mapping"]
        else:
            self.output_mapping = {}

    def evaluate(self,**kwargs):
        mapped_inputs = self.apply_input_mapping(kwargs)
        result = self.evaluate_mapped_inputs(**mapped_inputs)
        mapped_outputs = self.apply_output_mapping(result)
        return mapped_outputs

    def apply_input_mapping(self,inputs):
        mapped_inputs = {}
        for k,v in inputs.iteritems():
            if k in self.input_mapping:
                new_key = self.input_mapping[k]
                if new_key in self.input_mapping:
                    message = "input_mapping for '{}' would overwrite existing key.".format(k)
                    raise ValueError(message)
                mapped_inputs[new_key] = v
            else:
                if k in mapped_inputs:
                    message = "duplicate key '{}' found while mapping inputs.".format(k)
                    raise ValueError(message)
                mapped_inputs[k] = v
        return mapped_inputs

    def apply_output_mapping(self,outputs):
        mapped_outputs = {}
        for k,v in outputs.iteritems():
            if k in self.output_mapping:
                new_key = self.output_mapping[k]
                if new_key in self.output_mapping:
                    message = "output_mapping for '{}' would overwrite existing key.".format(k)
                    raise ValueError(message)
                mapped_outputs[new_key] = v
            else:
                if k in mapped_outputs:
                    message = "duplicate key '{}' found while mapping outputs.".format(k)
                    raise ValueError(message)
                mapped_outputs[k] = v
        return mapped_outputs

    def evaluate_mapped_inputs(self,**kwargs):
        raise NotImplementedError

class SequentialMeter(MeterBase):
    def __init__(self,sequence,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        assert all([issubclass(meter.__class__,MeterBase)
                    for meter in sequence])
        self.sequence = sequence

    def evaluate_mapped_inputs(self,**kwargs):
        result = {}
        for meter in self.sequence:
            args = {k:v for k,v in chain(kwargs.iteritems(),result.iteritems())}
            meter_result = meter.evaluate(**args)
            for k,v in meter_result.iteritems():
                if k in result:
                    message = "unexpected repeated metric ({}) in {}. " \
                              "A different input_mapping or " \
                              "output_mapping may fix this overlap.".format(k,meter)
                    raise ValueError(message)
                result[k] = v
        return result

class ConditionalMeter(MeterBase):
    def __init__(self,condition_parameter,success=None,failure=None,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.condition_parameter = condition_parameter
        self.success = success
        self.failure = failure

    def evaluate_mapped_inputs(self,**kwargs):
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

class TemperatureSensitivityParameterOptimizationMeter(MeterBase):
    def __init__(self,fuel_unit_str,fuel_type,temperature_unit_str,model,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.fuel_unit_str = fuel_unit_str
        self.fuel_type = fuel_type
        self.temperature_unit_str = temperature_unit_str
        self.model = model

    def evaluate_mapped_inputs(self,consumption_history,weather_source,**kwargs):
        consumptions = consumption_history.get(self.fuel_type)
        average_daily_usages = [c.average_daily_usage(self.fuel_unit_str) for c in consumptions]
        observed_daily_temps = weather_source.get_daily_temperatures(consumptions,self.temperature_unit_str)
        params = self.model.parameter_optimization(average_daily_usages,observed_daily_temps)
        return {"temp_sensitivity_params": params}

class AnnualizedUsageMeter(MeterBase):
    def __init__(self,temperature_unit_str,model,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.temperature_unit_str = temperature_unit_str
        self.model = model

    def evaluate_mapped_inputs(self,temp_sensitivity_params,weather_normal_source,**kwargs):
        daily_temps = weather_normal_source.annual_daily_temperatures(self.temperature_unit_str)
        usage_estimates = self.model.compute_usage_estimates(temp_sensitivity_params,daily_temps)
        annualized_usage = np.sum(usage_estimates)
        return {"annualized_usage":annualized_usage }

class PrePostMeter(MeterBase):
    def __init__(self,meter,splittable_args,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.meter = meter
        self.splittable_args = splittable_args

    def evaluate_mapped_inputs(self,retrofit_start_date,retrofit_end_date,**kwargs):
        pre_kwargs = {}
        post_kwargs = {}
        split_kwargs = {}
        for k,v in kwargs.iteritems():
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
        pre_results = {k + "_pre":v for k,v in pre_results.iteritems()}
        post_results = {k + "_post":v for k,v in post_results.iteritems()}
        results = {k:v for k,v in chain(pre_results.iteritems(),
                                        post_results.iteritems(),
                                        split_kwargs.iteritems())}
        return results

class GrossSavingsMeter(MeterBase):
    def __init__(self,model,fuel_unit_str,fuel_type,temperature_unit_str,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.model = model
        self.fuel_type = fuel_type
        self.fuel_unit_str = fuel_unit_str
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,temp_sensitivity_params_pre,consumption_history_post,weather_source,**kwargs):
        consumptions_post = consumption_history_post.get(self.fuel_type)
        observed_daily_temps = weather_source.get_daily_temperatures(consumptions_post,self.temperature_unit_str)
        usages = np.array([c.to(self.fuel_unit_str) for c in consumptions_post])
        usage_n_days = np.array([len(ts) for ts in observed_daily_temps])
        usage_estimates_post = np.array(self.model.compute_usage_estimates(temp_sensitivity_params_pre,observed_daily_temps)) * usage_n_days
        return {"gross_savings": np.sum(usages - usage_estimates_post)}

class AnnualizedGrossSavingsMeter(MeterBase):
    def __init__(self,model,fuel_type,temperature_unit_str,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.model = model
        self.fuel_type = fuel_type
        self.temperature_unit_str = temperature_unit_str

    def evaluate_mapped_inputs(self,temp_sensitivity_params_pre,temp_sensitivity_params_post,consumption_history_post,weather_normal_source,**kwargs):
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
        super(self.__class__,self).__init__(**kwargs)
        self.fuel_types = fuel_types

    def evaluate_mapped_inputs(self,consumption_history,**kwargs):
        results = {}
        for fuel_type in self.fuel_types:
            consumptions = consumption_history.get(fuel_type)
            results[fuel_type + "_presence"] = consumptions is not None
        return results

class DebugMeter(MeterBase):
    def evaluate_mapped_inputs(self,**kwargs):
        print "DEBUG kwargs:", kwargs
        return {}

class DummyMeter(MeterBase):
    def evaluate_mapped_inputs(self,**kwargs):
        result = {"result": kwargs["value"]}
        return result
