class Project(object):

    def __init__(self, trace_set, interventions,  location):
        self.interventions = interventions
        self.trace_set = trace_set
        self.location = location
