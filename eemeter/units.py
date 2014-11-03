# Unit conversion

class Unit:
    def __init__(self,name,abbr,description):
        self.name = name
        self.abbr = abbr
        self.description = description

    def __str__(self):
        return self.name

KilowattHour = Unit("KilowattHour","kWh","Unit of energy")
BritishThermalUnit = Unit("BritishThermalUnit","BTU","Unit of energy")
Therm = Unit("Therm","therm","Unit of energy")

def kwh_to_therm(kwh):
    """
    Return kwh value in therms
    """
    return kwh * 29.3001111

def therm_to_kwh(therm):
    """
    Return therm value in kwhs
    """
    return therm * 0.0341295634

def farenheight_to_celsius(f):
    """
    Return Farenheight value in Celsius
    """
    return (5./9) * (f-32)

def celsius_to_farenheight(c):
    """
    Return Celsius value in Farenheight
    """
    return (9./5)*c + 32

def temp_to_hdd(t,base):
    """
    Return temperature value (any units) in heating degree days (HDD_base)
    """
    if t >= base:
        return 0
    return base - t

def temp_to_cdd(t,base):
    """
    Return temperature value (any units) in cooling degree days (CDD_base)
    """
    if t <= base:
        return 0
    return t - base
