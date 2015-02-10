from .base import MeterBase

from datetime import datetime
from datetime import timedelta

class RecentReadingMeter(MeterBase):
    def __init__(self,n_days,**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.dt_target = datetime.now() - timedelta(days=n_days)

    def evaluate_mapped_inputs(self,consumption_history,**kwargs):
        recent_reading = False
        for consumption in consumption_history.iteritems():
            if consumption.end > self.dt_target:
                recent_reading = True
                break
        return {"recent_reading": recent_reading}

class EstimatedReadingConsolidationMeter(MeterBase):

    def evaluate_mapped_inputs(self,consumption_history,**kwargs):
        new_consumption_history = consumption_history
        return {"consumption_history": new_consumption_history}
