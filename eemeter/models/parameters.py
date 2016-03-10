import numpy as np
from copy import copy
import six

class ParameterMeta(type):
    def __new__(cls, name, parents, dct):
        dct["_parameter_names"] = dct["parameters"]
        del dct["parameters"]
        return super(ParameterMeta, cls).__new__(cls, name, parents, dct)

@six.add_metaclass(ParameterMeta)
class ParameterType(object):

    parameters = []

    def __init__(self, values):
        self._parameter_values = {}
        if type(values) == list or type(values) == np.ndarray:
            if not len(values) == len(self._parameter_names):
                message = "Values provided ({}) do not match parameter specification: {}" \
                        .format(values, self._parameter_names)
                raise TypeError(message)
            for name, value in zip(self._parameter_names, values):
                self._parameter_values[name] = value
        elif type(values) == dict:
            if not len(values) == len(self._parameter_names):
                message = "Values provided ({}) do not match parameter specification: {}" \
                        .format(values, self._parameter_names)
                raise TypeError(message)
            for name in self._parameter_names:
                self._parameter_values[name] = values[name]
        elif type(values) == type(self):
            self._parameter_names = copy(values._parameter_names)
            self._parameter_values = copy(values._parameter_values)
        else:
            message = "Should initialize with either a list or dictionary of parameter " \
                    "values, but got values={} instead".format(values)
            raise TypeError(message)

    def to_list(self):
        return [self._parameter_values[name] for name in self._parameter_names]

    def to_array(self):
        return np.array(self.to_list())

    def to_dict(self):
        return {name:self._parameter_values[name] for name in self._parameter_names}

    def json(self):
        return self.to_dict()
