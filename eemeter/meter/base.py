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

class PrePostMetricBase(MetricBase):
    def __init__(self,retrofit_start,retrofit_end):
        self.retrofit_start = retrofit_start
        self.retrofit_end = retrofit_end

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
        for key,value in dct.items():
            if issubclass(value.__class__,MetricBase):
                metric = value
                metrics[key] = metric
                metric_args = inspect.getargspec(metric.evaluate).args
                metric_args.pop(0)
                inputs[key] = metric_args

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
        data = {}
        for metric_name,metric in self.metrics.iteritems():
            inputs = self._inputs[metric_name]
            evaluation = metric.evaluate(*[kwargs[inpt] for inpt in inputs])
            data[metric_name] = evaluation
        return MeterRun(data)

