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
        for consumption in consumption_history.get(self.fuel_type):
            kWhs.append(consumption.to(self.unit_name))

        return np.mean(kWhs)

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

class FlagBase(MetricBase):

    def is_flag(self):
        return True

class FuelTypePresenceFlag(FlagBase):
    def __init__(self,fuel_type):
        self.fuel_type = fuel_type

    def evaluate(self,consumption_history):
        return len(consumption_history[self.fuel_type.name]) > 0

class NoneInTimeRangeFlag(FlagBase):
    pass

class OverlappingPeriodsFlag(FlagBase):
    pass

class MissingPeriodsFlag(FlagBase):
    pass

class TooManyEstimatedPeriodsFlag(FlagBase):
    pass

class ShortTimeSpanFlag(FlagBase):
    pass

class InsufficientTemperatureRangeFlag(FlagBase):
    pass

class MixedFuelTypeFlag(FlagBase):
    pass
