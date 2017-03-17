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

    def __init__(self, energy_trace_set, interventions, site, project_id=None):
        self.energy_trace_set = energy_trace_set
        self.interventions = interventions
        self.site = site
        self.project_id = project_id

    def __repr__(self):
        if self.project_id is not None:
            return "Project(project_id={})".format(self.project_id)
        return (
            "Project(energy_trace_set={}, interventions={}, site={}, project)"
            .format(self.energy_trace_set, self.interventions, self.site)
        )
