class FuelType:
    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

electricity = FuelType("Electricity")
natural_gas = FuelType("Natural gas")

class Usage:
    def __init__(self,usage,unit,fuel_type,start,end,estimated=False):
        self.usage = usage
        self.unit = unit
        self.fuel_type = fuel_type
        self.start = start
        self.end = end
        self.estimated = estimated

class UsageSet:
    def __init__(self):
        pass
