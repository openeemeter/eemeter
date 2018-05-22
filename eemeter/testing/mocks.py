import pandas as pd
import pytz
import numpy as np


class MockAVERTClient(object):

    def _fake_loads(self, n):
        # sinusoidal fake temperatures in degC
        avg_load = 25000.
        load_range = 10000.
        period_offset = - (2 * np.pi / 3)
        load_offsets = np.sin((2 * np.pi * np.arange(n) / n) + period_offset)
        return avg_load + (load_range * load_offsets)

    def read_rdf_file(self, year, region):
        dates = pd.date_range("{}-01-01 00:00".format(year),
                              "{}-12-31 23:00".format(year),
                              freq='D', tz=pytz.UTC)
        loads = self._fake_loads(dates.shape[0])
        co2_by_load = pd.Series(np.arange(0, 60000, 1000.),
                                np.arange(0, 60000, 1000.))
        return co2_by_load, pd.Series(loads, index=dates, dtype=float)
