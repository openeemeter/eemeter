from . import ureg, Q_

class DateRangeException(Exception): pass
class InvalidFuelTypeException(Exception): pass

class FuelType:
    """
    Class for representing fuel types
    """

    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

electricity = FuelType("Electricity")
natural_gas = FuelType("Natural gas")

class Consumption:
    """
    Class for representing energy usage
    """

    def __init__(self,usage,unit_name,fuel_type,start,end,estimated=False):
        self.fuel_type = fuel_type
        self.start = start
        self.end = end
        self.estimated = estimated

        quantity = usage * ureg.parse_expression(unit_name)

        self.joules = quantity.to(ureg.joules).magnitude

        if self.end < self.start:
            raise DateRangeException

        if not isinstance(self.fuel_type,FuelType):
            raise InvalidFuelTypeException

    def __repr__(self):
        return "Consumption: {} J".format(self.joules)

    def to(self,unit):
        """
        Return interally stored BTU value in the given unit
        """
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

    @property
    def timedelta(self):
        """
        Return the timedelta between the start and end datetimes
        """
        return self.end - self.start
