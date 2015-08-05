"""
Control-flow meters all override the evaluate function and execute meters
by directly passing the appropriate DataCollection object. Importantly, these
meters do not do the same input pre and post processing necessary in other
meters because the execute_raw function is not defined.

When defining meters in YAML, meters should be defined inline.
Custom meters can be included by referencing their names as accessible in the
__main__ module. (E.g. "!obj:__main__.MyMeter {...}").
"""

from .base import MeterBase
from eemeter.meter import DataCollection

class Sequence(MeterBase):
    """Collects and returns a series of meter object evaluation outputs in
    sequence, making the outputs of each meter available to those that
    follow.

    Parameters
    ----------
    sequence : list of eemeter.meter.MeterBase objects
        Meters will be executed in the order they are given; meters coming
        later in sequence will see the original inputs and the additional
        collected outputs from previously executed meters.
    """

    def __init__(self, sequence, **kwargs):
        super(Sequence,self).__init__(**kwargs)
        assert all([issubclass(meter.__class__, MeterBase)
                for meter in sequence])
        self.sequence = sequence

    def evaluate(self, data_collection):
        """Overrides normal execution to evaluate meters in sequence,
        collecting outputs.

        Returns
        -------
        out : dict
            Collected outputs from all meters in the sequence.
        """
        sequence_data_collection = data_collection.copy()
        for meter in self.sequence:
            meter_result = meter.evaluate(sequence_data_collection)
            sequence_data_collection.add_data_collection(meter_result)
        return sequence_data_collection

class Condition(MeterBase):
    """Collects and returns a series of meter object evaluation outputs in
    sequence, making the outputs of each meter available to those that
    follow.

    Parameters
    ----------
    condition : dict
        The search criteria for a parameter containing a boolean value.

        E.g.

            {
                "name": "input1",
                "tags": ["tag1", "tag2"]
            }

    success : eemeter.meter.MeterBase, optional
        The meter to execute if the condition is True.
    failure : eemeter.meter.MeterBase, optional
        The meter to execute if the condition is False.
    """

    def __init__(self, condition, success=None, failure=None, **kwargs):
        super(Condition, self).__init__(**kwargs)
        self.condition = condition
        self.success = success
        self.failure = failure

    def evaluate(self, data_collection):
        """Overrides the evalute method to evaluate either the `success` meter
        or the `failure` meter on the boolean value of a particular meter
        input stored in the element with the name `condition_parameter`.

        Returns
        -------
        out : dict
            Collected outputs from either the success or the failure meter.
        """
        if data_collection.get_data(**self.condition).value:
            meter = self.success
        else:
            meter = self.failure

        if meter is None:
            return DataCollection()
        else:
            return meter.evaluate(data_collection)

class Switch(MeterBase):
    """Switches between cases of values of a particular parameter.

    Parameters
    ----------
    target : dict
        The search criteria for a parameter on which to switch.

        E.g.

            {
                "name": "input1",
                "tags": ["tag1", "tag2"]
            }

    cases : dict
        A dictionary of meters to execute depending on the value of the target
        parameter; keyed on potential values of target.
    default : eemeter.meter.MeterBase, optional
        The meter to exectute if the value of the parameter was not found in
        the dictionary of cases.
    """
    def __init__(self, target, cases, default=None, **kwargs):
        super(Switch, self).__init__(**kwargs)
        self.target = target
        self.cases = cases
        self.default = default

    def evaluate(self, data_collection):
        """Determine the value of the target parameter and, according to that
        value, choose which meter to run.

        Returns
        -------
        out : dict
            Contains the results of the meter which was run.
        """
        item = data_collection.get_data(**self.target)
        if item is None:
            return DataCollection()
        meter = self.cases.get(item.value)
        if meter is not None:
            return meter.evaluate(data_collection)
        if self.default is not None:
            return self.default.evaluate(data_collection)
        return DataCollection()

class For(MeterBase):
    """ Operates like a python-style for loop, looping over parameters to feed
    to a meter.

    Parameters
    ----------
    variable_name : str
        Name for an object; will be appended to the auxiliary_inputs
        supplied to the meter on each iteration of the loop.
    iterable : dict
        Search criterion for an object over which to iterate. The iterable
        itself should be a list of dictionaries with the keys, "value" and
        "tags". The "tags" should be unique for each value.

        E.g.

            iterable = {
                "name": "consumption_data_all",
                "tags": ["tag1", "tag2"]
            }

        Which refers to an DataContainer like the following:

            {
                "name": "consumption_data_all",
                "value": [
                    {
                        "value": consumption_data,
                        "tags": ["electricity", "baseline"]
                    }, {
                         "value": consumption_data,
                         "tags": ["natural_gas", "baseline"]
                    }, {
                         "value": consumption_data,
                         "tags": ["electricity", "reporting"],
                    }, {
                         "value": consumption_data,
                         "tags": ["natural_gas", "reporting"]
                    }],
                "tags": ["tag1", "tag2"]
            }

    meter: eemeter.meter.MeterBase
        Meter to run on each iteration.

    """

    def __init__(self, variable_name, iterable, meter, **kwargs):
        super(For, self).__init__(**kwargs)
        self.variable_name = variable_name
        self.iterable = iterable
        self.meter = meter

    def evaluate(self, data_collection):
        """Determine the value of the target parameter and, according to that
        value, choose which meter to run.

        Returns
        -------
        out : dict
            Contains the results of each meter run.
        """


        original_aux_inputs = self.meter.auxiliary_inputs
        output_data_collection = DataCollection()
        for i in data_collection.get_data(**self.iterable).value:
            value = i.get("value")
            self.meter.auxiliary_inputs = dict(original_aux_inputs.items() + [(self.variable_name, value)])

            outputs = self.meter.evaluate(data_collection)

            tags = i.get("tags", [])
            outputs.add_tags(tags)
            output_data_collection.add_data_collection(outputs)
        return output_data_collection
