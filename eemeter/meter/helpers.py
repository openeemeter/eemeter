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
    values : list of str
        List of names of inputs for which to check acceptance criteria.
    thresholds : list of comparable items
        Thresholds that must be met. Must have same length as `values`.
    operations : list of {"lt","gt","lte","gte"}
        Direction of criterion. Options are less than, greater than, less than
        or equal to, or greater than or equal to. Must have same length as
        `values`.
    proportions : list of float
        Multipliers on the threshold (e.g. (value < threshold * proportion)).
        Must have same length as `values`.
    output_names : list of str
        Names of output booleans. Must have same length as `values`.
    """

    def __init__(self,values,thresholds,operations,proportions,output_names,**kwargs):
        super(MeetsThresholds,self).__init__(**kwargs)
        self.values = values
        self.thresholds = thresholds
        self.operations = operations
        self.proportions = proportions
        self.output_names = output_names

    def evaluate_mapped_inputs(self,**kwargs):
        """Evaluates threshold comparisons on incoming data.

        Parameters
        ----------
        out : dict
            Boolean outputs keyed on output names.
        """
        result = {}
        for v,t,o,p,n in zip(self.values,self.thresholds,self.operations,self.proportions,self.output_names):
            value = kwargs.get(v)
            if isinstance(t,basestring):
                threshold = kwargs.get(t)
            else:
                threshold = t
            if o == "lt":
                result[n] = (value < threshold * p)
            elif o == "gt":
                result[n] = (value > threshold * p)
            elif o == "lte":
                result[n] = (value <= threshold * p)
            elif o == "gte":
                result[n] = (value >= threshold * p)
        return result

class EstimatedReadingConsolidationMeter(MeterBase):
    """Consolidates estimated readings by either combining them with actual
    reads or dropping them entirely (e.g. final read is estimated).
    """

    def evaluate_mapped_inputs(self,consumption_data,**kwargs):
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
    def evaluate_mapped_inputs(self,**kwargs):
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
    def evaluate_mapped_inputs(self, value, **kwargs):
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
