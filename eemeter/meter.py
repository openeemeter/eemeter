import numpy as np

class MetricBase:
    def evaluate(self,consumption_history):
        raise NotImplementedError

class RawAverageUsageMetric(MetricBase):
    def __init__(self,fuel_type,unit_name):
        self.fuel_type = fuel_type
        self.unit_name = unit_name

    def evaluate(self,consumption_history):
        kWhs = []
        for consumption in consumption_history.get(self.fuel_type):
            kWhs.append(consumption.to(self.unit_name))

        return np.mean(kWhs)

class Meter:
    pass

class FlagBase:

    def __init__(self,raised):
        self.raised = raised

    def description(self):
        raise NotImplementedError

class NoneInTimeRangeFlag(FlagBase):

    def description(self):
        return "None in time range"

class OverlappingPeriodsFlag(FlagBase):

    def description(self):
        return "Overlapping time periods"

class MissingPeriodsFlag(FlagBase):

    def description(self):
        return "Missing time periods"

class TooManyEstimatedPeriodsFlag(FlagBase):

    def __init__(self,raised,limit):
        self.raised = raised
        self.limit = limit

    def description(self):
        return "More than {} estimated periods".format(self.limit)

class ShortTimeSpanFlag(FlagBase):

    def __init__(self,raised,limit):
        self.raised = raised
        self.limit = limit

    def description(self):
        return "Fewer than {} days in sample".format(self.limit)

class InsufficientTemperatureRangeFlag(FlagBase):

    def description(self):
        return "Insufficient temperature range"

class MixedFuelTypeFlag(FlagBase):

    def description(self):
        return "Mixed fuel types"

