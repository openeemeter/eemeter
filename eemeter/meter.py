import numpy as np

class MetricBase:
    def evaluate(self,consumption_history):
        raise NotImplementedError

    def is_flag(self):
        return False

class RawAverageUsageMetric(MetricBase):
    def __init__(self,fuel_type,unit_name):
        self.fuel_type = fuel_type
        self.unit_name = unit_name

    def evaluate(self,consumption_history):
        kWhs = []
        consumptions = consumption_history.get(self.fuel_type)
        if consumptions:
            for consumption in consumptions:
                kWhs.append(consumption.to(self.unit_name))
        return np.mean(kWhs)

class FlagBase(MetricBase):

    def is_flag(self):
        return True

class FuelTypePresenceFlag(FlagBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history):
        return consumption_history.get(self.fuel_type.name) is not None

class TimeRangePresenceFlag(FlagBase):
    def __init__(self,start,end):
        assert start <= end
        self.start = start
        self.end = end

    def evaluate(self,consumption_history):
        for consumption in consumption_history.iteritems():
            print consumption
            if self._in_time_range(consumption.start) or \
                self._in_time_range(consumption.end):
                return True
        return False

    def _in_time_range(self,dt):
        return self.start <= dt and self.end >= dt

class OverlappingTimePeriodsFlag(FlagBase):
    def evaluate(self,consumption_history):
        for fuel_type,consumptions in consumption_history.fuel_types():
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
    def evaluate(self,consumption_history):
        for fuel_type,consumptions in consumption_history.fuel_types():
            consumptions.sort()
            if len(consumptions) <= 1:
                return False
            for c1,c2 in zip(consumptions,consumptions[1:]):
                if c1.end != c2.start:
                    return True
        return False

class TooManyEstimatedPeriodsFlag(FlagBase):
    def __init__(self,maximum):
        self.maximum = maximum

    def evaluate(self,consumption_history):
        for fuel_type,consumptions in consumption_history.fuel_types():
            num_estimated = len([c for c in consumptions if c.estimated])
            if num_estimated > self.maximum:
                return True
        return False

class InsufficientTimeRangeFlag(FlagBase):
    def __init__(self,days):
        self.days = days

    def evaluate(self,consumption_history):
        for fuel_type,consumptions in consumption_history.fuel_types():
            consumptions.sort()
            if (consumptions[-1].end - consumptions[0].start).days < self.days:
                return True
        return False

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

