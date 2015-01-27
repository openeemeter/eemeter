from consumption import Consumption
from eemeter.consumption import electricity

class ConsumptionGenerator:
    def __init__(self, fuel_type, consumption_unit_name, weather_unit_name, heat_base, heat_sensitivity, cool_base, cool_sensitivity):
        self.fuel_type = fuel_type
        self.consumption_unit_name = consumption_unit_name
        self.weather_unit_name = weather_unit_name

        self.heat_base = heat_base
        self.heat_sensitivity = cool_sensitivity

        self.cool_base = heat_base
        self.cool_sensitivity = cool_sensitivity

    # noise is an instance of scipy.stats.rv_continuous, e.g. scipy.stats.normal()
    # noise is additive and sampled independently for each period
    def generate(self, weather_source, periods, noise = None):
        hdds = weather_source.get_hdd(periods, self.weather_unit_name, self.heat_base)
        cdds = weather_source.get_cdd(periods, self.weather_unit_name, self.cool_base)

        consumptions = []
        for i in range(len(periods)):
            u = hdds[i]*self.heat_sensitivity + cdds[i]*self.cool_sensitivity
            
            if noise != None:
                u += noise.rvs()
            
            c = Consumption(u, self.consumption_unit_name, self.fuel_type, periods[i].start, periods[i].end)
            consumptions.append(c)

        return consumptions