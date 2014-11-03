class DateRangeException(Exception):
    pass

class InvalidFuelTypeException(Exception):
    pass

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

class Usage:
    """
    Class for representing energy usage
    """

    def __init__(self,usage,unit,fuel_type,start,end,estimated=False):
        self.fuel_type = fuel_type
        self.start = start
        self.end = end
        self.estimated = estimated

        self.BTU = unit.toBTU(usage)

        if self.end < self.start:
            raise DateRangeException

        if not isinstance(self.fuel_type,FuelType):
            raise InvalidFuelTypeException

    def to(self,unit):
        """
        Return interally stored BTU value in the given unit
        """
        return unit.fromBTU(self.BTU)

    @property
    def timedelta(self):
        """
        Return the timedelta between the start and end datetimes
        """
        return self.end - self.start
