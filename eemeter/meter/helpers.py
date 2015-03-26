from .base import MeterBase

from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory

from itertools import chain
from pprint import pprint

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str,bytes)

class PrePost(MeterBase):
    """Meter which divides data into pre- and post- retrofit parts, then
    executes potentially different pre- and post- retrofit meters.

    Parameters
    ----------
    pre_meter : eemeter.meter.MeterBase
        Meter to evaluate on pre-retrofit data. Meter results keys will be
        prepended with "_pre". (Note, it will have access to both pre- and
        post-retrofit data.
    post_meter : eemeter.meter.MeterBase
        Meter to evaluate on pre-retrofit data. Meter results keys will be
        prepended with "_post". (Note: it will have access to both pre- and
        post-retrofit data.
    splittable_args : list of str
        List of keys which should be split using the `before(date)` and
        `after(date)` methods.
    """

    def __init__(self,pre_meter,post_meter,splittable_args,**kwargs):
        super(PrePost,self).__init__(**kwargs)
        self.pre_meter = pre_meter
        self.post_meter = post_meter
        self.splittable_args = splittable_args

    def evaluate_mapped_inputs(self,retrofit_start_date,retrofit_completion_date,**kwargs):
        """Splits consuption_history into pre and post retrofit periods, then
        evaluates a meter on each subset of consumptions, appending the strings
        `"_pre"` and `"_post"`, respectively, to each key of each meter output.
        The values "is_pre" and "is_post" are also defined.

        Parameters
        ----------
        retrofit_start_date : datetime.date or datetime.datetime
            Date of retrofit start. "pre" data will include consumptions which
            end before or on this date.
        retrofit_completion_date : datetime.date or datetime.datetime
            Date of retrofit start. "post" data will include consumptions which
            begin after or on this date.

        Returns
        -------
        out : dict
            Dictionary with results of pre- and post- retrofit meter results
            and split inputs. Results will have "_pre" or "_post" appended to
            key string.
        """

        pre_kwargs = {
            "is_pre":True,
            "is_post":False,
            "retrofit_start_date": retrofit_start_date,
            "retrofit_completion_date": retrofit_completion_date
        }
        post_kwargs = {
            "is_pre": False,
            "is_post": True,
            "retrofit_start_date": retrofit_start_date,
            "retrofit_completion_date": retrofit_completion_date
        }
        split_kwargs = {}
        for k,v in kwargs.items():
            if k in self.splittable_args:
                pre_kwargs[k] = v.before(retrofit_start_date)
                post_kwargs[k] = v.after(retrofit_completion_date)
                split_kwargs[k + "_pre"] = pre_kwargs[k]
                split_kwargs[k + "_post"] = post_kwargs[k]
            else:
                pre_kwargs[k] = v
                post_kwargs[k] = v
        pre_results = self.pre_meter.evaluate(**pre_kwargs)
        post_results = self.post_meter.evaluate(**post_kwargs)
        pre_results = {k + "_pre":v for k,v in pre_results.items()}
        post_results = {k + "_post":v for k,v in post_results.items()}
        results = {k:v for k,v in chain(pre_results.items(),
                                        post_results.items(),
                                        split_kwargs.items())}
        return results

class MeetsThresholds(MeterBase):
    """Evaluates whether or not particular named metrics meet thresholds of
    acceptance criteria and returns booleans indicating acceptance or failure.

    Parameters
    ----------
    values : list of str
        List of names of inputs for which to check acceptance criteria.
    thresholds : list of comparable items
        Thresholds that must be met. Must have same length as `values`.
    operations : list of {"lt","gt","lte","gte"}
        Direction of criterion. Options are less than, greater than, less than
        or equal to, or greater than or equal to. Must have same length as
        `values`.
    proportions : list of float
        Multipliers on the threshold (e.g. (value < threshold * proportion)).
        Must have same length as `values`.
    output_names : list of str
        Names of output booleans. Must have same length as `values`.
    """

    def __init__(self,values,thresholds,operations,proportions,output_names,**kwargs):
        super(MeetsThresholds,self).__init__(**kwargs)
        self.values = values
        self.thresholds = thresholds
        self.operations = operations
        self.proportions = proportions
        self.output_names = output_names

    def evaluate_mapped_inputs(self,**kwargs):
        """Evaluates threshold comparisons on incoming data.

        Parameters
        ----------
        out : dict
            Boolean outputs keyed on output names.
        """
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
    """Consolidates estimated readings by either combining them with actual
    reads or dropping them entirely (e.g. final read is estimated).
    """

    def evaluate_mapped_inputs(self,consumption_history,**kwargs):
        """Evaluates threshold comparisons on incoming data.

        Parameters
        ----------
        consumption_history : eemeter.consumption.ConsumptionHistory
            Meter readings to consolidate.

        Returns
        -------
        out : dict
            Contains the consolidated consumption history keyed by the string
            "consumption_history_no_estimated".
        """
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

        Returns
        -------
        out : {}
        """
        print("DEBUG")
        pprint(kwargs)
        return {}

class DummyMeter(MeterBase):
    def evaluate_mapped_inputs(self,value,**kwargs):
        """Helpful for testing meters or creating simple pass through meters.

        Parameters
        ----------
        value : object
            Value to return

        Returns
        -------
        out : dict
            Value stored on key "result".
        """
        result = {"result": value}
        return result
