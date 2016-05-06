import inspect
from collections import defaultdict
from six import string_types
import json
import numpy as np
from eemeter.config.yaml_parser import load

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

        if isinstance(tags, string_types):
            message = "tags should be a list or None, got tags={}".format(tags)
            raise TypeError(message)

        if tags is None:
            self.tags = frozenset()
        else:
            self.tags = frozenset(tags)

    def __repr__(self):
        return "<{}:{} (tags={})>".format(self.name, self.value, list(self.tags))

    def set_value(self, value):
        """ Set the value.
        """
        self.value = value

    def get_value(self, json_serializable=False):
        """ Retrieve the value
        """
        if json_serializable:
            try:
                json.dumps(self.value)
                return self.value
            except TypeError as e:
                try:
                    if type(self.value) == list: # for iterables
                        return [{
                            "value": v["value"].json(),
                            "tags": v["tags"],
                        } for v in self.value]
                    else: # for values explicitly defining json method, e.g. parameters
                        return self.value.json()
                except AttributeError:
                    try: # for numpy arrays
                        return self.value.tolist()
                    except AttributeError:
                        raise e
                    raise e
        else:
            return self.value

    def add_tags(self, tags):
        """ Add tags to the container.
        """

        if isinstance(tags, string_types):
            message = "tags should be a list or None, got tags={}".format(tags)
            raise TypeError(message)

        self.tags = self.tags.union(tags)


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

        if isinstance(tags, string_types):
            message = "tags should be a list or None, got tags={}".format(tags)
            raise TypeError(message)

        self._name_index = defaultdict(list)
        for name, value in kwargs.items():
            data_container = DataContainer(name=name, value=value, tags=tags)
            self.add_data(data_container)

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
        if isinstance(tagspace, string_types):
            message = "tagspace should be a list or None, got tags={}".format(tagspace)
            raise TypeError(message)

        for item in data_collection.iteritems():
            new_item = DataContainer(item.name, item.value, item.tags | set(tagspace))
            self.add_data(new_item)

    def add_tags(self, tags):
        """ Add a set of tags to each object in the collection.
        """

        if isinstance(tags, string_types):
            message = "tags should be a list or None, got tags={}".format(tags)
            raise TypeError(message)

        for item in self.iteritems():
            item.add_tags(tags)

    def get_data(self, name, tags=None):
        """ Retrieve a single item (or None, if none exists) matching the name
        and tag set supplied.
        """

        if isinstance(tags, string_types):
            message = "tags should be a list or None, got tags={}".format(tags)
            raise TypeError(message)

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
        for name, data_containers in self._name_index.items():
            for data_container in data_containers:
                yield data_container

    def json(self):
        """Serializes data. Non-serializable outputs are replaced with
        "NOT_SERIALIZABLE".
        """
        json_data = {}

        for item in self.iteritems():
            try:
                value = item.get_value(json_serializable=True)
            except TypeError:
                value = "NOT_SERIALIZABLE"
            item_data = {
                "tags": list(item.tags),
                "value": value,
            }
            if item.name in json_data:
                json_data[item.name].append(item_data)
            else:
                json_data[item.name] = [item_data]

        return json_data

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

        # collect raw strings
        item_names, item_values, item_tags = [], [], []
        for item in self.iteritems():
            item_names.append( item.name)
            item_values.append("\n".join([
                "    {}".format(l)
                for l in "{!s}".format(item.value).splitlines()
            ]))
            item_tags.append("{}".format(sorted(list(item.tags))))

        # length of longest name will be used for padding
        max_len = max([len(i) for i in item_names])

        # construct total string
        for name, value, tags in zip(item_names, item_values, item_tags):
            line = (
                "\n{name:-<{padding}}{tags}\n\n{value}\n"
                .format(name=name, tags=tags, value=value, padding=max_len + 3)
            )
            string += line
        return string

    def search(self, string, tags=None):
        """ Returns any data containers matching the search criteria.

        Parameters
        ----------

        string : str
            Criteria for matching container names.
        tags : list of str, default None
            Matches only if one of the tags provided here also matches.

        Returns
        -------
        data_collection : list of eemeter.meter.DataContainer
            Matching items; unordered.
        """

        if isinstance(tags, string_types):
            message = "tags should be a list or None, got tags={}".format(tags)
            raise TypeError(message)

        data_collection = DataCollection()
        for item in self.iteritems():
            if string == item.name:
                if tags is None or tags == []:
                    data_collection.add_data(item)
                else:
                    if any([tag in item.tags for tag in tags]):
                        data_collection.add_data(item)
        return data_collection

    def filter_by_tag(self, tags):
        """ Returns any data containers matching the filter criteria.

        Parameters
        ----------

        tags : list of str, default None
            Matches only if all of the tags provided here also match.

        Returns
        -------
        data_collection : DataCollection
            Matching items.
        """

        if isinstance(tags, string_types):
            message = "tags should be a list or None, got tags={}".format(tags)
            raise TypeError(message)

        data_collection = DataCollection()
        for item in self.iteritems():
            if tags == [] or tags == None or all([tag in item.tags for tag in tags]):
                data_collection.add_data(item)
        return data_collection

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
                results = {"output_name1": output1, "output_name2": output2}
                return results

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
        all_inputs = mapped_input_dict.copy()
        all_inputs.update(self.auxiliary_inputs)

        # evaluate meter
        outputs_dict = self.evaluate_raw(**all_inputs)

        # combine auxiliary outputs with raw outputs
        all_outputs = outputs_dict.copy()
        all_outputs.update(self.auxiliary_outputs)


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
        for target_name, search_criteria in mapping.items():
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
        for result_name, target_data in mapping.items():

            target_value = data_dict.get(result_name)
            if target_value is None:
                message = "Data not found during mapping: {}" \
                        .format(result_name)
                raise ValueError(message)

            if type(target_data) is not list:
                target_data = [target_data]
            for td in target_data:
                target_name = td.get("name", result_name)
                target_tags = td.get("tags", [])
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

    def yaml_mapping(self):
        args = inspect.getargspec(self.__init__).args[1:]
        mapping = { arg: getattr(self, arg) for arg in args}
        mapping["input_mapping"] = self.input_mapping
        mapping["output_mapping"] = self.output_mapping
        mapping["auxiliary_inputs"] = self.auxiliary_inputs
        mapping["auxiliary_outputs"] = self.auxiliary_outputs
        mapping["tagspace"] = self.tagspace
        return mapping

class YamlDefinedMeter(MeterBase):
    """Meter type which uses yaml internally.
    """

    def __init__(self, settings={}, **kwargs):
        super(YamlDefinedMeter, self).__init__(**kwargs)
        self.settings = self.process_settings(settings)
        self.meter = load(self.yaml, self.settings)

    @property
    def yaml(self):
        """Should return a string with the raw yaml.
        """
        raise NotImplementedError

    def process_settings(self, settings):
        """ Processes settings by adding defaults where needed and eliminating
        unused settings.

        Parameters
        ----------
        settings : dict
            Settings dict where each key-value pair is a name and a value.

        Returns
        -------
        processed_settings : dict
            Settings dict where each key-value pair is a name and a value;
            defaults are provided where necessary. Contains exactly the keys
            in the default settings dict.

        """
        default_settings = self.default_settings()

        processed_settings = {}

        for key, value in default_settings.items():
            if key in settings:
                processed_settings[key] = settings[key]
            else:
                processed_settings[key] = value

        self.validate_settings(processed_settings)

        return processed_settings

    def default_settings(self, settings):
        """Use this method to provide default settings.
        """
        return {}

    def validate_settings(self, settings):
        """Use this method to provide extra validation for settings.
        Raise errors if the settings are not valid.
        """
        pass

    def evaluate_raw(self):
        """This method is not used in YamlDefinedMeters.
        """
        message = "Use the meter.evaluate(data_collection) method for" \
                "YamlDefinedMeters."
        raise NotImplementedError(message)

    def evaluate(self, data_collection):
        """Evaluates the meter defined by yaml. Requires that a meter
        property exist on the meter after initialization.
        """
        outputs = self.meter.evaluate(data_collection)
        outputs.add_tags(self.tagspace)
        return outputs

    def yaml_mapping(self):
        mapping = super(YamlDefinedMeter, self).yaml_mapping()
        mapping["meter"] = self.meter
        return mapping
