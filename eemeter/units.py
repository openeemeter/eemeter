class Unit:
    def __init__(self,name,abbr,description):
        self.name = name
        self.abbr = abbr
        self.description = description

    def __str__(self):
        return self.name

kWh = Unit("KilowattHour","kWh","Unit of energy")
BTU = Unit("BritishThermalUnit","BTU","Unit of energy")
therm = Unit("Therm","therm","Unit of energy")
