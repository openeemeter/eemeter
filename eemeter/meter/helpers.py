from .base import MeterBase

from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory

from itertools import chain

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str,bytes)

class PrePost(MeterBase):
    def __init__(self,pre_meter,post_meter,splittable_args,**kwargs):
        super(PrePost,self).__init__(**kwargs)
        self.pre_meter = pre_meter
        self.post_meter = post_meter
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
        pre_results = self.pre_meter.evaluate(**pre_kwargs)
        post_results = self.post_meter.evaluate(**post_kwargs)
        pre_results = {k + "_pre":v for k,v in pre_results.items()}
        post_results = {k + "_post":v for k,v in post_results.items()}
        results = {k:v for k,v in chain(pre_results.items(),
                                        post_results.items(),
                                        split_kwargs.items())}
        return results

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

class EstimatedReadingConsolidationMeter(MeterBase):

    def evaluate_mapped_inputs(self,consumption_history,**kwargs):
        def combine_waitlist(wl):
            usage = sum([c.to("kWh") for c in wl])
            ft = wl[0].fuel_type
            return Consumption(usage, "kWh", ft, wl[0].start, wl[-1].end,estimated=False)

        new_consumptions = []
        for fuel_type,consumptions in consumption_history.fuel_types():
            waitlist = []
            for c in sorted(consumptions):
                waitlist.append(c)
                if not c.estimated:
                    new_consumptions.append(combine_waitlist(waitlist))
                    waitlist = []

        return {"consumption_history_no_estimated": ConsumptionHistory(new_consumptions)}

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
