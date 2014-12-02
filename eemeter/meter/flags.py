from .base import FlagBase

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
