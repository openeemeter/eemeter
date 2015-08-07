import inspect
from collections import defaultdict

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str,bytes)

class DataContainer:
    """ Data structure for tagged data. Associates a name and tag with a
    particular value. Forms the basis of the DataCollection.
    """

    def __init__(self, name, value, tags=None):
        self.name = name
        self.set_value(value)
        if tags is None:
            self.tags = frozenset()
        else:
            self.tags = frozenset(tags)

    def set_value(self, value):
        """ Set the value.
        """
        self.value = value

    def get_value(self):
        """ Retrieve the value
        """
        return self.value

    def add_tags(self, tags):
        """ Add tags to the container.
        """
        self.tags = self.tags.union(tags)

    def __repr__(self):
        return "<{}:{} (tags={})>".format(self.name, self.value, list(self.tags))

class DataCollection:
    """ Stores and allows retrieval of multiple tagged and named data objects.
    """

    def __init__(self, tags=[], **kwargs):
        """ Allows initialization using key-value pairs as keyword arguments.


        E.g.:

            data_collection = DataCollection(a="value_a", b="value_b", tags=["tag1"])

        This object will now hold two data containers, each with a single tag
        ("tag1"). The two data containers will have, respectively, the names
        "a" and "b", with the values "value_a" and "value_b". Values can be any
        python object.

        """
        self._name_index = defaultdict(list)
        for k,v in kwargs.iteritems():
            dc = DataContainer(name=k, value=v, tags=tags)
            self.add_data(dc)

    def add_data(self, data_container):
        """ Add tagged, named data to the collection.
        """
        name, tags = data_container.name, data_container.tags
        existing_element = self.get_data(name, tags)
        if existing_element is None:
            self._name_index[data_container.name].append(data_container)
        else:
            message = "Element already exists in data container: {}" \
                    .format(existing_element)
            raise ValueError(message)

    def add_data_collection(self, data_collection, tagspace=[]):
        """ Add an entire data collection to the data, optionally with a set of
        tags to apply before adding.
        """
        for item in data_collection.iteritems():
            new_item = DataContainer(item.name, item.value, item.tags | set(tagspace))
            self.add_data(new_item)

    def add_tags(self, tags):
        """ Add a set of tags to each object in the collection.
        """
        for item in self.iteritems():
            item.add_tags(tags)

    def get_data(self, name, tags=None):
        """ Retrieve a single item (or None, if none exists) matching the name
        and tag set supplied.
        """
        potential_matches = self._name_index[name]
        if tags is None:
            matches = potential_matches
        else:
            matches = []
            for potential_match in potential_matches:
                is_match = all(tag in potential_match.tags for tag in tags)
                if is_match:
                    matches.append(potential_match)
        n_matches = len(matches)
        if n_matches == 0:
            return None
        elif n_matches == 1:
            return matches[0]
        else:
            message = "Ambiguous criteria: found {} matches for" \
                    " name={}, tags={}".format(n_matches, name, tags)
            raise ValueError(message)

    def iteritems(self):
        """ Iterate through values in the data collection.
        """
        for name, data_containers in self._name_index.iteritems():
            for data_container in data_containers:
                yield data_container

    def copy(self):
        """ Create a new DataCollection containing the same objects and tags.
        """
        new_data_collection = DataCollection()
        for item in self.iteritems():
            new_data_collection.add_data(item)
        return new_data_collection

    def count(self):
        """ Returns the number of stored data objects.
        """
        return len([i for i in self.iteritems()])

    def __repr__(self):
        string = "DataCollection ({} items)".format(self.count())
        for item in self.iteritems():
            string += "\n  {:>30}  {:<30} tags={}".format(item.name, item.value, list(item.tags))
        return string

class MeterBase(object):
    """Base class for all Meter objects. Takes care of structural tasks such as
    input and output mapping.

    Parameters
    ----------
    input_mapping : dict, optional
        Maps inputs to evaluate_raw (keys) from DataCollection object search
        criteria (values).

        Example::

            {
                "input_name1": {
                    "name": "name1",
                    "tags": ["tag0"]
                },
                "input_name2": {
                    "name": "name2",
                    "tags": ["tag0"]
                },
            }

        This input_mapping could be used to map objects in a DataCollection
        that looks like this::

            data_collection = DataCollection(name1="value1", name2="value2", tags=["tag0"])

        to inputs to an evaluate_raw function that looks like this::

            def evaluate_raw(self, input_name1, input_name2):
                ...

        Behind the scenes, the data will be converted into a format like this
        for evaluation::

            input_args = {"input_name1": "value1", "input_name2": "value2"}
            self.evaluate_raw(**input_args)

    output_mapping : dict, optional
        Maps result dictionary outputs from evaluate_raw to an output
        DataCollection.

        Example::

            {
                "output_name1": {
                    "name": "name1",
                    "tags": ["tag1"]
                },
                "output_name2": {
                    "name": "name2",
                    "tags": ["tag2"]
                },
            }

        After a call to evaluate_raw resulting in an output that looked like
        this::

            results = {
                "output_name1": "result1",
                "output_name2": "result2",
            }

        The output_mapping shown above would result in a DataCollection with
        the following elements::

            DataCollection (2 items)
                name1  "result1"    tags=["tag1"]
                name2  "result2"    tags=["tag2"]

    auxiliary_inputs : dict
        Extra key/value pairs to make available in the meter evaluation
        namespace. These will be added *after* input mapping has occurred.

        Example::

            {
                "extra_input1": "value1",
                "extra_input2": "value2"
            }


    auxiliary_outputs : dict
        Extra key/value pairs to add to the output namespace. Note that these
        will be added *before* output mapping has occurred, so they will also
        need to be explicitly added to the output space.

        Example::

            {
                "extra_output1": "value1",
                "extra_output2": "value2"
            }


    tagspace : list of str
        Tags to apply to outputs generated local to this meter.

        Example::

            ["tag_for_all_submeters"]

    """

    def __init__(self, input_mapping={}, output_mapping={},
            auxiliary_inputs={}, auxiliary_outputs={}, tagspace={}, **kwargs):
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.auxiliary_inputs = auxiliary_inputs
        self.auxiliary_outputs = auxiliary_outputs
        self.tagspace = tagspace

    def evaluate(self, data_collection):
        """ The preferred method for evaluating meters. Handles input mapping,
        intermediate value collection, and computed output export.

        Parameters
        ----------
        data_collection : eemeter.meter.DataCollection
            Contains data needed for meter evaluation.
        """
        # map inputs
        mapped_input_dict = self._dict_from_data_collection(self.input_mapping,
                data_collection)

        # combine auxiliary inputs with mapped inputs
        all_inputs = dict(mapped_input_dict.items() + self.auxiliary_inputs.items())

        # evaluate meter
        outputs_dict = self.evaluate_raw(**all_inputs)

        # combine auxiliary outputs with raw outputs
        all_outputs = dict(outputs_dict.items() + self.auxiliary_outputs.items())


        # map meter evaluations back to data_collection form
        mapped_output_data_collection = self._data_collection_from_dict(
                self.output_mapping, all_outputs)

        # combine with original data, add tags as necessary
        mapped_output_data_collection.add_tags(self.tagspace)

        return mapped_output_data_collection

    def _dict_from_data_collection(self, mapping, data_collection):
        """ Resolves a DataCollection to dict mapping.
        """
        data_dict = {}
        for target_name, search_criteria in mapping.iteritems():
            search_name = search_criteria.get("name", target_name)
            search_tags = search_criteria.get("tags", [])
            target_data = data_collection.get_data(search_name, search_tags)
            if target_data is None:
                message = "Data not found during mapping: name={}, tags={}" \
                        .format(search_name, search_tags)
                raise ValueError(message)
            else:
                data_dict[target_name] = target_data.get_value()
        return data_dict

    def _data_collection_from_dict(self, mapping, data_dict):
        """ Resolves a dict to DataCollection mapping.
        """
        data_collection = DataCollection()
        for result_name, target_data in mapping.iteritems():
            target_name = target_data.get("name", result_name)
            target_tags = target_data.get("tags", [])
            target_value = data_dict.get(result_name)
            if target_value is None:
                message = "Data not found during mapping: {}" \
                        .format(result_name)
                raise ValueError(message)
            else:
                data = DataContainer(target_name, target_value, target_tags)
                data_collection.add_data(data)
        return data_collection

    def evaluate_raw(self):
        """Should be the workhorse method which implements the logic of the
        meter, returning a dictionary of meter outputs. Must be defined by
        inheriting class.

        This function will be called after inputs are mapped from a
        DataCollection using an input_mapping.

        The convention is that evaluate_raw should return a dictionary of
        results in order to be compatible with the output_mapping attribute,
        and other meters.
        """
        raise NotImplementedError
