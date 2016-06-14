class Project(object):
    ''' Container for storing project data.

    Parameters
    ----------
    trace_set : eemeter.structures.TraceSet
        Complete set of energy traces for this project. For a project site that
        has, for example, two electricity meters, each with two traces
        (supplied electricity kWh, and solar-generated kWh) and one natural gas
        meter with one trace (consumed natural gas therms), the `trace_set`
        should contain 5 traces, regardless of the availablity of that data.
        Traces which are unavailable should be represented as
        'placeholder' traces.
    interventions : list of eemeter.structures.Intervention
        Complete set of interventions, planned, ongoing, or completed,
        that have taken or will take place at this site as part of this
        project.
    site : eemeter.structures.Site
        The site of this project.
    '''

    def __init__(self, trace_set, interventions, site):
        self.trace_set = trace_set
        self.interventions = interventions
        self.site = site
