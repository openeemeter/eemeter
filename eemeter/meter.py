import numpy as np
from scipy import stats
from .consumption import FuelType
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

class RawAverageUsageMetric(MetricBase):
    def __init__(self,unit_name,fuel_type=None):
        # TODO - allow different units for different fuel types.
        self.unit_name = unit_name
        super(RawAverageUsageMetric,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
        """Returns the average usage with the specified unit and the specified
        fuel type.
        """
        if consumptions is None:
            return np.nan
        return np.mean([consumption.to(self.unit_name) for consumption in consumptions])

class TemperatureRegressionParametersMetric(MetricBase):

    # TODO - weight these by likelyhood.
    balance_points = range(55,70)

    def __init__(self,unit_name,fuel_type,weather_getter):
        self.fuel_type = fuel_type
        self.weather_getter = weather_getter

    def evaluate(self,consumption_history):
        usages = [c.to("kWh") for c in consumption_history.get(self.fuel_type)]
        avg_temps = self.weather_getter.get_average_temperature(consumption_history,self.fuel_type)
        best_coeffs = None,None
        best_r_value = -np.inf
        for balance_point in self.balance_points:
            u,t = self._filter_by_balance_point(balance_point,usages,avg_temps)
            slope, intercept, r_value, p_value, std_err = stats.linregress(u,t)
            if r_value > best_r_value and not np.isnan(p_value):
                best_coeffs = slope,intercept
        return best_coeffs

    @staticmethod
    def _filter_by_balance_point(balance_point,usages,avg_temps):
        data = [(usage,avg_temp) for usage,avg_temp in zip(usages,avg_temps) if avg_temp >= balance_point]
        if data:
            return zip(*data)
        else:
            return [],[]

class FlagBase(MetricBase):
    def __init__(self,fuel_type=None):
        if fuel_type:
            assert isinstance(fuel_type,FuelType)
        self.fuel_type = fuel_type

    def is_flag(self):
        """Returns `True` if the metric is a flag, and `False` otherwise."""
        return True

class FuelTypePresenceFlag(FlagBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history):
        """Returns True if the specified fuel_type is present in the
        specified consumption history.
        """
        return consumption_history.get(self.fuel_type) is not None

class TimeRangePresenceFlag(FlagBase):
    def __init__(self,start,end,fuel_type=None):
        assert start <= end
        self.start = start
        self.end = end
        super(TimeRangePresenceFlag,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
        """Returns `True` if any consumption of a particular fuel type is
        present in the specified time range.
        """
        if consumptions is None:
            return False
        for consumption in consumptions:
            if self._in_time_range(consumption.start) or \
                self._in_time_range(consumption.end):
                return True
        return False

    def _in_time_range(self,dt):
        """Returns `True` if `dt` is in the specified time range.
        """
        return self.start <= dt and self.end >= dt

class OverlappingTimePeriodsFlag(FlagBase):
    def evaluate_fuel_type(self,consumptions):
        """Returns `True` if any consumptions of the specified fuel types
        have overlapping time periods.
        """
        if consumptions is None:
            return False
        consumptions.sort()
        if len(consumptions) <= 1:
            return False
        for c1,c2 in zip(consumptions,consumptions[1:]):
            if c1.end == c2.end:
                if not (c1.start == c1.end or c2.start == c2.end):
                    return True
            elif c1.start == c2.start:
                if not (c1.start == c1.end or c2.start == c2.end):
                    return True
            else:
                if c2.start < c1.end:
                    return True
        return False


class MissingTimePeriodsFlag(FlagBase):
    def evaluate_fuel_type(self,consumptions):
        """Returns `True` if, taken in total, the consumption history of a
        particular fuel type, is missing any time periods.
        """
        if consumptions is None:
            return False
        consumptions.sort()
        if len(consumptions) <= 1:
            return False
        for c1,c2 in zip(consumptions,consumptions[1:]):
            if c1.end != c2.start:
                return True
        return False

class TooManyEstimatedPeriodsFlag(FlagBase):
    def __init__(self,maximum,fuel_type=None):
        self.maximum = maximum
        super(TooManyEstimatedPeriodsFlag,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
        """Returns `True` if the consumption history of a particular fuel type
        has more than the maximum allowable number of estimated readings.
        """
        if consumptions is None:
            return False
        return len([c for c in consumptions if c.estimated]) > self.maximum

class InsufficientTimeRangeFlag(FlagBase):
    def __init__(self,days,fuel_type=None):
        self.days = days
        super(InsufficientTimeRangeFlag,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
        """Returns `True` if the consumption history of a particular fuel type,
        fails to have a range at least as large as the number of days
        specified.
        """
        if consumptions is None:
            return True
        consumptions.sort()
        return (consumptions[-1].end - consumptions[0].start).days < self.days

class InsufficientTemperatureRangeFlag(FlagBase):
    pass

class MeterRun:
    def __init__(self,data):
        self._data = data

    def __getattr__(self,attr):
        return self._data[attr]

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

