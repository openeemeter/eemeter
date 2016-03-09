from .base import MeterBase

from eemeter.consumption import ConsumptionData

from itertools import chain
from pprint import pprint

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str,bytes)

class MeetsThresholds(MeterBase):
    """Evaluates whether or not particular named metrics meet thresholds of
    acceptance criteria and returns booleans indicating acceptance or failure.

    Parameters
    ----------
    equations : list of lists
        Each list should contain 6 elements:

        [value, inequality, proportion, threshold, bias, output_name]

        E.g.

            ["input1", "<", 2, "input2", 0, "output_name1"]

        Is roughly equivalent to:

            output_name1 = bool(input1 < 3 * input2 + 0)

        - output_name : str
            Name of output booleans.
        - value : str
            Name of input for which to evaluate acceptance criteria.
        - inequality : {"<",">","<=",">="}
            Inequality to use during evaluation.
        - proportion, threshold, bias: float, int, or str
            number or name of real-valued input
    """

    def __init__(self, equations, **kwargs):
        super(MeetsThresholds,self).__init__(**kwargs)
        self.equations = equations
        for e in equations:
            assert len(e) == 6

    def evaluate_raw(self, **kwargs):
        """Evaluates threshold comparisons on incoming data.

        Parameters
        ----------
        out : dict
            Boolean outputs keyed on output names.
        """
        result = {}
        for v,i,p,t,b,n in self.equations:
            value = kwargs.get(v)
            p = kwargs.get(p) if isinstance(p, basestring) else float(p)
            t = kwargs.get(t) if isinstance(t, basestring) else float(t)
            b = kwargs.get(b) if isinstance(b, basestring) else float(b)
            if i == "<":
                result[n] = bool(value < p*t + b)
            elif i == ">":
                result[n] = bool(value > p*t + b)
            elif i == "<=":
                result[n] = bool(value <= p*t + b)
            elif i == ">=":
                result[n] = bool(value >= p*t + b)
            else:
                message = "Inequality not recognized: {}".format(i)
                raise ValueError(message)
        return result

class EstimatedReadingConsolidationMeter(MeterBase):
    """Consolidates estimated readings by either combining them with actual
    reads or dropping them entirely (e.g. final read is estimated).
    """

    def evaluate_raw(self, consumption_data, **kwargs):
        """Evaluates threshold comparisons on incoming data.

        Parameters
        ----------
        consumption_data : eemeter.consumption.ConsumptionData
            Meter readings to consolidate.

        Returns
        -------
        out : dict
            Contains the consolidated consumption data keyed by the string
            "consumption_data_no_estimated".
        """
        def combine_records(records):
            return {"start": records[0]["start"], "value":sum([r["value"] for r in records])}

        values = consumption_data.data
        index = values.index
        estimated = consumption_data.estimated

        new_records = []
        record_waitlist = []
        for i, v, e in zip(index,values,estimated):
            record_waitlist.append({"start": i,"value": v})
            if not e:
                new_records.append(combine_records(record_waitlist))
                record_waitlist = []
        cd_no_est = ConsumptionData(new_records,consumption_data.fuel_type,
                consumption_data.unit_name, record_type="arbitrary_start")
        return {"consumption_data_no_estimated": cd_no_est}

class Debug(MeterBase):
    def evaluate_raw(self,**kwargs):
        """Helpful for debugging meter instances - prints out kwargs for
        inspection.

        Returns
        -------
        out : {}
        """
        print("DEBUG")
        pprint(kwargs)
        return {}

class DummyMeter(MeterBase):
    def evaluate_raw(self, value, **kwargs):
        """Helpful for testing meters or creating simple pass through meters.

        Parameters
        ----------
        value : object
            Value to return

        Returns
        -------
        out : dict
            Value stored on key "result".
        """
        result = {"result": value}
        return result
