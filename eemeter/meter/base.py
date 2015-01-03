from ..consumption import FuelType

import inspect

class MetricBase(object):
    def __init__(self,fuel_type=None):
        if fuel_type:
            assert isinstance(fuel_type,FuelType)
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history):
        """Evaluates the metric on the specified fuel_type or on every
        available fuel type, if none is specified, returning a dictionary of
        the evaluations keyed on fuel_type name. Requires specification of the
        `evaluate_fuel_type` method.
        """
        if self.fuel_type is None:
            usages = {}
            for fuel_type,consumptions in consumption_history.fuel_types():
                usages[fuel_type] = self.evaluate_fuel_type(consumptions)
            return usages
        else:
            consumptions = consumption_history.get(self.fuel_type)
            return self.evaluate_fuel_type(consumptions)

    def evaluate_fuel_type(self,consumptions):
        """Must be overridden by subclasses. Should return a value representing
        the metric as applied to consumption history of a particular fuel type.
        """
        raise NotImplementedError

    def is_flag(self):
        """Returns `True` if the metric is a flag, and `False` otherwise.
        """
        return False

class PrePost(MetricBase):
    def __init__(self,metric):
        self.metric = metric

    def evaluate(self,*args,**kwargs):
        """Returns a dictionary with keys "pre" and "post", with values
        corresponding to metric evaluation on pre or post data. For metric
        arguments to  be split pre/post, they must respond to "before" or
        "after" methods.
        """
        args_pre,args_post = self._split_args(args,kwargs)
        kwargs_pre,kwargs_post = self._split_kwargs(kwargs)
        pre = self.metric.evaluate(*args_pre,**kwargs_pre)
        post = self.metric.evaluate(*args_post,**kwargs_post)
        return {"pre": pre,"post": post}

    def _split_args(self,args,kwargs):
        args_pre = []
        args_post = []
        for arg in args:
            try:
                args_pre.append(arg.before(kwargs["retrofit_start"]))
                args_post.append(arg.after(kwargs["retrofit_end"]))
            except KeyError:
                args_pre.append(arg)
                args_post.append(arg)
        return args_pre, args_post

    def _split_kwargs(self,kwargs):
        kwargs_pre = {}
        kwargs_post = {}
        for key,arg in kwargs.iteritems():
            if not (key == "retrofit_start" or key == "retrofit_end"):
                try:
                    kwargs_pre[key] = arg.before(kwargs["retrofit_start"])
                    kwargs_post[key] = arg.after(kwargs["retrofit_end"])
                except KeyError:
                    kwargs_pre[key] = arg
                    kwargs_post[key] = arg
        return kwargs_pre, kwargs_post

class FlagBase(MetricBase):
    def __init__(self,fuel_type=None):
        if fuel_type:
            assert isinstance(fuel_type,FuelType)
        self.fuel_type = fuel_type

    def is_flag(self):
        """Returns `True` if the metric is a flag, and `False` otherwise."""
        return True

class MeterRun:
    def __init__(self,data):
        self._data = data

    def __getattr__(self,attr):
        return self._data[attr]

    def __str__(self):
        return "MeterRun({})".format(self._data)

class MeterMeta(type):
    def __new__(cls, name, parents, dct):
        metrics = {}
        inputs = {}

        def _extract_arguments(metric):
            metric_args = inspect.getargspec(metric.evaluate).args
            metric_args.pop(0)
            return metric_args

        # gather metrics and metric arguments
        for attr_name,attr in dct.items():
            if issubclass(attr.__class__,MetricBase):
                metrics[attr_name] = attr
                inputs[attr_name] = _extract_arguments(attr)

        if "stages" in dct:
            for stage in dct["stages"]:
                for attr_name,attr in stage.iteritems():
                    inputs[attr_name] = _extract_arguments(attr)

        dct["metrics"] = metrics
        dct["_inputs"] = inputs

        return super(MeterMeta, cls).__new__(cls, name, parents, dct)

class Meter(object):
    """All new meters should subclass the Meter class in order to have all of
    the expected functionality.
    """

    __metaclass__ = MeterMeta

    def run(self,**kwargs):
        """Returns a dictionary of the evaluations of all of the metrics and
        flags in this Meter, keyed by the names of the metric attributes
        supplied.
        """
        # first compute metrics not part of a particular stage
        data = {}
        for metric_name,metric in self.metrics.iteritems():
            inputs = self._inputs[metric_name]
            evaluation = metric.evaluate(*[kwargs[inpt] for inpt in inputs])
            data[metric_name] = evaluation

        # next, compute metrics stage by stage, if any
        if hasattr(self,"stages"):
            for stage in self.stages:
                for metric_name,metric in stage.iteritems():
                    args = self._get_arguments(data,kwargs,metric_name)
                    evaluation = metric.evaluate(*args)
                    data[metric_name] = evaluation

        return MeterRun(data)

    def _get_arguments(self,data,kwargs,metric_name):
        inputs = self._inputs[metric_name]
        args = []
        for inpt in inputs:
            if inpt in kwargs:
                args.append(kwargs[inpt])
            elif inpt in data:
                args.append(data[inpt])
            else:
                raise ValueError("Could not find matching input for {}".format(metric_name))
        return args
