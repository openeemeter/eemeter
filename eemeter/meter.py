class BaseFlag:

    def __init__(self,raised):
        self.raised = raised

    def description(self):
        raise NotImplementedError

class NoneInTimeRangeFlag(BaseFlag):

    def description(self):
        return "None in time range"

class OverlappingPeriodsFlag(BaseFlag):

    def description(self):
        return "Overlapping time periods"

class MissingPeriodsFlag(BaseFlag):

    def description(self):
        return "Missing time periods"

class TooManyEstimatedPeriodsFlag(BaseFlag):

    def __init__(self,raised,limit):
        self.raised = raised
        self.limit = limit

    def description(self):
        return "More than {} estimated periods".format(self.limit)

class ShortTimeSpanFlag(BaseFlag):

    def __init__(self,raised,limit):
        self.raised = raised
        self.limit = limit

    def description(self):
        return "Fewer than {} days in sample".format(self.limit)

class InsufficientTemperatureRangeFlag(BaseFlag):

    def description(self):
        return "Insufficient temperature range"

class MixedFuelTypeFlag(BaseFlag):

    def description(self):
        return "Mixed fuel types"

