import numpy as np
from .consumption import FuelType

class MetricBase(object):
    def __init__(self,fuel_type=None):
        if fuel_type:
            assert isinstance(fuel_type,FuelType)
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history):
        if self.fuel_type is None:
            usages = {}
            for fuel_type,consumptions in consumption_history.fuel_types():
                usages[fuel_type] = self.evaluate_fuel_type(consumptions)
            return usages
        else:
            consumptions = consumption_history.get(self.fuel_type)
            return self.evaluate_fuel_type(consumptions)

    def evaluate_fuel_type(self,consumptions):
        raise NotImplementedError

    def is_flag(self):
        return False

class RawAverageUsageMetric(MetricBase):
    def __init__(self,unit_name,fuel_type=None):
        self.unit_name = unit_name
        super(RawAverageUsageMetric,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
        if consumptions is None:
            return np.nan
        return np.mean([consumption.to(self.unit_name) for consumption in consumptions])

class FlagBase(MetricBase):
    def __init__(self,fuel_type=None):
        if fuel_type:
            assert isinstance(fuel_type,FuelType)
        self.fuel_type = fuel_type

    def is_flag(self):
        return True

class FuelTypePresenceFlag(FlagBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history):
        return consumption_history.get(self.fuel_type) is not None

class TimeRangePresenceFlag(FlagBase):
    def __init__(self,start,end,fuel_type=None):
        assert start <= end
        self.start = start
        self.end = end
        super(TimeRangePresenceFlag,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
        if consumptions is None:
            return False
        for consumption in consumptions:
            if self._in_time_range(consumption.start) or \
                self._in_time_range(consumption.end):
                return True
        return False

    def _in_time_range(self,dt):
        return self.start <= dt and self.end >= dt

class OverlappingTimePeriodsFlag(FlagBase):
    def evaluate_fuel_type(self,consumptions):
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
        if consumptions is None:
            return False
        return len([c for c in consumptions if c.estimated]) > self.maximum

class InsufficientTimeRangeFlag(FlagBase):
    def __init__(self,days,fuel_type=None):
        self.days = days
        super(InsufficientTimeRangeFlag,self).__init__(fuel_type)

    def evaluate_fuel_type(self,consumptions):
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
        for key,value in dct.items():
            if issubclass(value.__class__,MetricBase):
                metrics[key] = value

        dct["metrics"] = metrics

        return super(MeterMeta, cls).__new__(cls, name, parents, dct)

class Meter(object):

    __metaclass__ = MeterMeta

    def run(self,consumption_history):
        data = {}
        for metric_name,metric in self.metrics.iteritems():
            evaluation = metric.evaluate(consumption_history)
            data[metric_name] = evaluation
        return MeterRun(data)

