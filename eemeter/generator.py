from consumption import Consumption
import itertools

class ConsumptionGenerator:
    def __init__(self, fuel_type, consumption_unit_name, weather_unit_name, heat_base, heat_sensitivity, cool_base, cool_sensitivity, daily_base_load):
        self.fuel_type = fuel_type
        self.consumption_unit_name = consumption_unit_name
        self.weather_unit_name = weather_unit_name

        self.heat_base = heat_base
        self.heat_sensitivity = cool_sensitivity

        self.cool_base = heat_base
        self.cool_sensitivity = cool_sensitivity

        self.daily_base_load = daily_base_load

    # noise is an instance of scipy.stats.rv_continuous, e.g. scipy.stats.normal()
    # noise is additive and sampled independently for each period
    def generate(self, weather_source, periods, noise = None):
        hdds = weather_source.get_hdd(periods, self.weather_unit_name, self.heat_base)
        cdds = weather_source.get_cdd(periods, self.weather_unit_name, self.cool_base)

        consumptions = []
        for hdd,cdd,period in zip(hdds,cdds,periods):
            u = hdd*self.heat_sensitivity + cdd*self.cool_sensitivity

            if noise != None:
                u += noise.rvs()

            u += self.daily_base_load * period.timedelta.days

            c = Consumption(u, self.consumption_unit_name, self.fuel_type, period.start, period.end)
            consumptions.append(c)

        return consumptions
    

class ProjectGenerator:
    def __init__(self, name, n_homes, fuel_type, consumption_unit_name, weather_unit_name, 
               heat_base_dist, heat_sensitivity_dist, 
               cool_base_dist, cool_sensitivity_dist,
               daily_base_load_dist):
        
        self.name = name
        self.n_homes = n_homes
        self.fuel_type = fuel_type
        self.consumption_unit_name = consumption_unit_name
        
        self.homes = []
        
        for i in range(n_homes):
            c = ConsumptionGenerator(fuel_type, consumption_unit_name, weather_unit_name, 
                                     heat_base_dist.rvs(), heat_sensitivity_dist.rvs(), 
                                     cool_base_dist.rvs(), cool_sensitivity_dist.rvs(), 
                                     daily_base_load_dist.rvs())
            self.homes.append(c)
            
    def generate(self, weather_source, periods, noise=None):
        return [h.generate(weather_source, periods, noise) for h in self.homes]
    
class PortfolioGenerator:
    def __init__(self, name, projects):
        self.name = name
        self.projects = projects
    
    def generate(self, weather_sources, periods, noises=None):
        if noises is None: noises = [None for p in projects]
        return list(itertools.chain.from_iterable(
            g.generate(w,p,n) for g,w,p,n in zip(self.projects, weather_sources, periods, noises)
        ))