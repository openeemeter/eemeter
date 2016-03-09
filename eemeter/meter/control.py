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
from eemeter.meter import DataContainer

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
        input_data_collection = data_collection.copy()
        output_data_collection = DataCollection()
        for meter in self.sequence:
            meter_result = meter.evaluate(input_data_collection)
            input_data_collection.add_data_collection(meter_result)
            output_data_collection.add_data_collection(meter_result)
        return output_data_collection

class Condition(MeterBase):
    """Collects and returns a series of meter object evaluation outputs in
    sequence, making the outputs of each meter available to those that
    follow.

    Parameters
    ----------
    condition : dict
        The search criteria for a parameter containing a boolean value.

        Example::

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
        """Overrides the evaluate method to evaluate either the `success` meter
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

        Example::

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
        if meter is None:
            if self.default is None:
                return DataCollection()
            else:
                meter = self.default
        output_data_collection = meter.evaluate(data_collection)
        return output_data_collection

class For(MeterBase):
    """ Operates like a python-style for loop, looping over parameters to feed
    to a meter.

    Parameters
    ----------
    variable : str
        Name for an object; will be appended to the data collection
        supplied to the meter on each iteration of the loop.

        Example::

            {
                "name": "consumption_data",
                "tags": ["tag1", "tag2"]
            }

    iterable : dict
        Search criterion for an object over which to iterate. The iterable
        itself should be a list of dictionaries with the keys, "value" and
        "tags". The "tags" should be unique for each value.

        Example::

            {
                "name": "consumption_data_all",
                "tags": ["tag1", "tag2"]
            }

        Which refers to an DataContainer structured like the following::

            {
                "name": "consumption_data_all",
                "value": [
                    {
                        "value": consumption_data1,
                        "tags": ["electricity", "baseline"]
                    },
                    {
                         "value": consumption_data2,
                         "tags": ["natural_gas", "baseline"]
                    },
                    {
                         "value": consumption_data3,
                         "tags": ["electricity", "reporting"],
                    },
                    {
                         "value": consumption_data4,
                         "tags": ["natural_gas", "reporting"]
                    }
                ],
                "tags": ["tag1", "tag2"]
            }

    meter: eemeter.meter.MeterBase
        Meter to run on each iteration.

    """

    def __init__(self, variable, iterable, meter, **kwargs):
        super(For, self).__init__(**kwargs)
        self.variable = variable
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
        output_data_collection = DataCollection()
        for i in data_collection.get_data(**self.iterable).value:
            variable_name = self.variable["name"]
            variable_value = i.get("value")
            variable_tags = self.variable.get("tags",[])
            data_container = DataContainer(
                    name=variable_name,
                    value=variable_value,
                    tags=variable_tags)

            input_data_collection = data_collection.copy()
            input_data_collection.add_data(data_container)

            outputs = self.meter.evaluate(input_data_collection)

            tags = i.get("tags", [])
            outputs.add_tags(tags)
            output_data_collection.add_data_collection(outputs)
        return output_data_collection

class TagFilter(MeterBase):
    """ Filters input data to match tag criteria.

    Parameters
    ----------
    meter: eemeter.meter.MeterBase
        Meter to run on each iteration.
    """

    def __init__(self, meter, **kwargs):
        super(TagFilter, self).__init__(**kwargs)
        self.meter = meter

    def get_tags(self, data_collection):
        """
        Should return a list of tags to be used in a filter_by_tag call
        on the input data_collection.

        Parameters
        ----------
        data_collection : eemeter.meter.DataCollection
            Input data from which to pull the relevant tags.


        Returns
        -------

        tags : list
            Tags to filter by.
        """
        raise NotImplementedError

    def evaluate(self, data_collection):
        """Filter meter inputs according to some search criteria and execute
        the meter provided during initialization.

        Additional input data not satisfying the filter criteria may be
        retained by adding it to the input_mapping.

        Returns
        -------
        out : eemeter.meter.DataCollection
            Contains the results of the meter run.
        """

        # prepare input data
        data_filter = self.get_tags(data_collection)
        input_data_collection = data_collection.filter_by_tag(data_filter)
        mapped_input_dict = self._dict_from_data_collection(self.input_mapping,
                data_collection)
        for name, value in mapped_input_dict.items():
            input_data_collection.add_data(DataContainer(name, value))

        # execute the meter
        output_data_collection = self.meter.evaluate(input_data_collection)

        return output_data_collection

class FuelTypeTagFilter(TagFilter):

    def __init__(self, fuel_type_search_name, **kwargs):
        super(FuelTypeTagFilter, self).__init__(**kwargs)
        self.fuel_type_search_name = fuel_type_search_name

    def get_tags(self, data_collection):
        """
        Should return the current fuel type, which will be used in a
        filter_by_tag call on the input data_collection.

        Parameters
        ----------
        data_collection : eemeter.meter.DataCollection
            Input data from which to pull the relevant tags.

        Returns
        -------

        tags : list
            Tags to filter by.
        """
        tags = [data_collection.get_data(self.fuel_type_search_name).value]
        return tags

