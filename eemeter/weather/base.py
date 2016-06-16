from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.core.common import is_list_like


class WeatherSourceBase(object):

    def __init__(self, station):
        self.station = station
        self.tempC = pd.Series(dtype=float)

    @staticmethod
    def _unit_convert(x, unit):
        if unit is None or unit == "degC":
            return x
        elif unit == "degF":
            return 1.8*x + 32
        else:
            message = (
                "Unit not supported ({}). Use 'degF' or 'degC'"
                .format(unit)
            )
            raise NotImplementedError(message)

    def json(self):
        return {
            "station": self.station,
            "records": [{
                "datetime": d.strftime(self.cache_date_format),
                "tempC": t if pd.notnull(t) else None,
            } for d, t in self.tempC.iteritems()]
        }
