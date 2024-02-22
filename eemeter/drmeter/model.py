from eemeter.eemeter.models.hourly import HourlyModel


class Model(HourlyModel):
    def __init__(self, settings=None):
        self.segment_type = "single"
        self.alpha = 0.1

    