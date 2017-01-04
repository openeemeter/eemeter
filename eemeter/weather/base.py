import pandas as pd


class WeatherSourceBase(object):

    def __init__(self, station):
        self.station = station
        self.tempC = pd.Series(dtype=float)

    @staticmethod
    def _unit_convert(x, unit):
        if unit is None or unit == "degC":
            return x
        elif unit == "degF":
            return 1.8 * x + 32
        else:
            message = (
                "Unit not supported ({}). Use 'degF' or 'degC'"
                .format(unit)
            )
            raise ValueError(message)
