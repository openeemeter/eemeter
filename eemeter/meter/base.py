import inspect
from collections import defaultdict

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str,bytes)

class DataContainer:

    def __init__(self, name, value, tags=None):
        self.name = name
        self.set_value(value)
        if tags is None:
            self.tags = frozenset()
        else:
            self.tags = frozenset(tags)

    def set_value(self,value):
        self.value = value

    def get_value(self):
        return self.value

class DataCollection:

    def __init__(self):
        self._name_index = defaultdict(list)

    def add_data(self, data_container):
        name, tags = data_container.name, data_container.tags
        existing_element = self.get_data(name, tags)
        if existing_element is None:
            self._name_index[data_container.name].append(data_container)
        else:
            message = "Element already exists in data container: {}" \
                    .format(existing_element)
            raise ValueError(message)

    def add_data_collection(self, data_collection, tagspace=[]):
        for item in data_collection.iteritems():
            new_item = DataContainer(item.name, item.value, item.tags | set(tagspace))
            self.add_data(new_item)

    def get_data(self, name, tags=None):
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
        for name, data_containers in self._name_index.iteritems():
            for data_container in data_containers:
                yield data_container

    def copy(self):
        new_data_collection = DataCollection()
        for item in self.iteritems():
            new_data_collection.add_data(item)
        return new_data_collection

class MeterBase(object):
    """Base class for all Meter objects. Takes care of structural tasks such as
    input and output mapping.

    Parameters
    ----------
    input_mapping, output_mapping : dict, optional
        Keys are incoming input (or output) names and values are outgoing
        input (or output) names. To map one key to multiple values, use a
        list; to remove a key, use None. **Note:** in YAML, `None` is spelled
        `null`; otherwise, it will be interpreted as a string.

        E.g.

            input_mapping = {
                "input_name1": {
                    "name": "data_collection_name1",
                    "tags": []
                },
                "input_name2": {
                    "name": "data_collection_name2",
                    "tags": []
                },
            }

    auxiliary_data : dict
        Extra key/value pairs to make available in the meter evaluation
        namespace.

        E.g.

        auxiliary_data = {
            "extra_input1": {
                "value": value,
                "tags": ["tag1","tag2"],
            },
            "extra_input2": {
                "value": value,
                "tags": ["tag1","tag2"],
            },
        }
    tagspace : list of str
        Tags to apply to outputs generated local to this meter.

        E.g.

        tagspace = ["tag_for_all_submeters"]
    """
    def __init__(self, input_mapping={}, output_mapping={}, auxiliary_data={},
            tagspace={}, **kwargs):
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.auxiliary_data = auxiliary_data
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

        # combine aux data with mapped inputs
        all_inputs = dict(mapped_input_dict.items() + self.auxiliary_data.items())

        # evaluate meter
        result_dict = self.evaluate_mapped_inputs(**all_inputs)

        # map meter evaluations back to data_collection form
        mapped_output_data_collection = self._data_collection_from_dict(
                self.output_mapping, result_dict)

        # combine with original data, adding tags as necessary
        output_data_collection = data_collection.copy()
        output_data_collection.add_data_collection(
                mapped_output_data_collection, self.tagspace)

        return output_data_collection

    def _dict_from_data_collection(self, mapping, data_collection):
        """ Resolves a DataCollection to dict mapping.
        """
        data_dict = {}
        for target_name, search_criteria in mapping.iteritems():
            search_name = search_criteria.get("name", target_name)
            search_tags = search_criteria.get("tags", [])
            target_data = data_collection.get_data(search_name, search_tags)
            if target_data is None:
                message = "Data not found during mapping: name={}, tags=" \
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

    def evaluate_raw(self, **kwargs):
        """Should be the workhorse method which implements the logic of the
        meter, returning a dictionary of meter outputs. Must be defined by
        inheriting class.
        """
        raise NotImplementedError
