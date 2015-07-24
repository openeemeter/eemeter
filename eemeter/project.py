class Project(object):
    """
    Parameters
    ----------
    location : eemeter.location.Location
        Location object representing the location of the building in the
        project.
    consumption : list of eemeter.consumption.ConsumptionData objects
        All available consumption data for this project.
    baseline_period : eemeter.evaluation.Period object
        Date/time period for baselining.
    reporting_period : eemeter.evaluation.Period object
        Date/time period for reporting.
    other_periods : list of eemeter.evaluation.Period objects
        Other named date/time periods of interest, perhaps particular seasons or years of interest.
    """

    def __init__(self,location,consumption=[],baseline_period=None,reporting_period=None,other_periods=[]):
        self.location = location
        self.consumption = consumption
        self.baseline_period = baseline_period
        self.reporting_period = reporting_period
        self.other_periods = other_periods
