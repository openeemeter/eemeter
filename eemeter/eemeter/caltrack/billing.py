from eemeter.eemeter.models import DailyModel

class BillingModel(DailyModel): 
    """wrapper for DailyModel using billing presets"""
    def __init__(self, settings=None):
        super().__init__(model="2.0", settings=settings)

    def _initialize_data(self, meter_data):
        #TODO add typehint + checks for meter_data once dataclass is up
        return super()._initialize_data(meter_data)