from . import ureg, Q_
from collections import defaultdict

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

electricity = FuelType("electricity")
natural_gas = FuelType("natural_gas")
propane = FuelType("propane")

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

    def __getattr__(self,unit):
        """
        Return interally stored joules value in the given unit
        """
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

    def to(self,unit):
        """
        Return interally stored joules value in the given unit
        """
        return (self.joules * ureg.joules).to(ureg.parse_expression(unit)).magnitude

    @property
    def timedelta(self):
        """
        Return the timedelta between the start and end datetimes
        """
        return self.end - self.start

class ConsumptionHistory:
    def __init__(self,consumptions):
        self._data = defaultdict(list)
        for consumption in consumptions:
            self._data[consumption.fuel_type.name].append(consumption)

    def __getattr__(self, attr):
        return self._data[attr]

    def __repr__(self):
        return "ConsumptionHistory({})".format(self._data)

    def get(self,fuel_type):
        return self._data[fuel_type.name]
