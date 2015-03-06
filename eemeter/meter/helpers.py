from .base import MeterBase
from eemeter.consumption import Consumption
from eemeter.consumption import ConsumptionHistory

from datetime import datetime
from datetime import timedelta

class RecentReadingMeter(MeterBase):
    def __init__(self,n_days,since_date=datetime.now(),**kwargs):
        super(self.__class__,self).__init__(**kwargs)
        self.dt_target = since_date - timedelta(days=n_days)

    def evaluate_mapped_inputs(self,consumption_history,fuel_type,**kwargs):
        recent_reading = False
        for consumption in consumption_history.get(fuel_type):
            if consumption.end > self.dt_target:
                recent_reading = True
                break
        return {"recent_reading": recent_reading}

class EstimatedReadingConsolidationMeter(MeterBase):

    def evaluate_mapped_inputs(self,consumption_history,**kwargs):
        def combine_waitlist(wl):
            usage = sum([c.to("kWh") for c in wl])
            ft = wl[0].fuel_type
            return Consumption(usage, "kWh", ft, wl[0].start, wl[-1].end,estimated=False)

        new_consumptions = []
        for fuel_type,consumptions in consumption_history.fuel_types():
            waitlist = []
            for c in sorted(consumptions):
                waitlist.append(c)
                if not c.estimated:
                    new_consumptions.append(combine_waitlist(waitlist))
                    waitlist = []

        return {"consumption_history_no_estimated": ConsumptionHistory(new_consumptions)}
