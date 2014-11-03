class FuelType:
    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

electricity = FuelType("Electricity")
natural_gas = FuelType("Natural gas")

class Usage:
    def __init__(self,usage,unit,fuel_type,start,end,estimated=False):
        self.fuel_type = fuel_type
        self.start = start
        self.end = end
        self.estimated = estimated

        self.BTU = unit.toBTU(usage)

    def to(self,unit):
        return unit.fromBTU(self.BTU)
