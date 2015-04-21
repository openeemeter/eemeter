import inspect

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    basestring = (str,bytes)

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

        E.g. ({"old_input_name":"new_input_name"})
    extras : dict
        Extra persistent key/value pairs to make available in the meter
        evaluation namespace.
    """
    def __init__(self,input_mapping={},output_mapping={},extras={},**kwargs):
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.extras = extras

    def evaluate(self,**kwargs):
        """Thin wrapper on `evaluate_mapped_inputs` which performs input
        and output mappings. Arguments must be specified as keyword arguments.
        This is the preferred method for evaluating meters.
        """
        mapped_inputs = self._remap(kwargs,self.input_mapping)
        for k,v in self.extras.items():
            if k in mapped_inputs:
                message = "duplicate key '{}' found while adding extras to inputs"\
                        .format(k)
                raise ValueError(message)
            else:
                mapped_inputs[k] = v
        inputs = self.get_inputs()[self.__class__.__name__]["inputs"]
        optional_inputs = self._get_optional_inputs()
        for inpt in inputs:
            if inpt not in mapped_inputs and inpt not in optional_inputs:
                message = "expected argument '{}' for meter '{}'; "\
                          "got kwargs={} (with mapped_inputs={}) instead."\
                                  .format(inpt,self.__class__.__name__,
                                          sorted(kwargs.keys()),sorted(mapped_inputs.keys()))
                raise TypeError(message)
        result = self.evaluate_mapped_inputs(**mapped_inputs)
        mapped_outputs = self._remap(result,self.output_mapping)
        for k,v in self.extras.items():
            if k in mapped_outputs:
                message = "duplicate key '{}' found while adding extras to inputs"\
                        .format(k)
                raise ValueError(message)
            else:
                mapped_outputs[k] = v
        return mapped_outputs

    @staticmethod
    def _remap(inputs,mapping):
        """Returns a dictionary with mapped keys. `mapping` should be a
        dictionary with incoming input names as keys whose associated values
        are outgoing input names. (e.g. ({"old_key":"new_key"})
        """
        mapped_dict = {}
        for k,v in inputs.items():
            mapped_keys = mapping.get(k,k)
            if mapped_keys is not None:
                if isinstance(mapped_keys, basestring):
                    mapped_keys = [mapped_keys]
                for mapped_key in mapped_keys:
                    if mapped_key in mapped_dict:
                        message = "duplicate key '{}' found while mapping"\
                                .format(mapped_key)
                        raise ValueError(message)
                    mapped_dict[mapped_key] = v
        return mapped_dict

    def evaluate_mapped_inputs(self,**kwargs):
        """Should be the workhorse method which implements the logic of the
        meter, returning a dictionary of meter outputs. Must be defined by
        inheriting class.
        """
        raise NotImplementedError

    def get_inputs(self):
        """A structured object which shows necessary inputs.

        Returns
        -------
        out : dict
            Structure mirroring the structure of the meter and showing its
            inputs and child_inputs.
        """
        inputs = inspect.getargspec(self.evaluate_mapped_inputs).args[1:]
        child_inputs = self._get_child_inputs()
        return {self.__class__.__name__:{"inputs":inputs,"child_inputs":child_inputs}}

    def _get_optional_inputs(self):
        argspec = inspect.getargspec(self.evaluate_mapped_inputs)
        if argspec.defaults is None:
            return {}
        return dict(zip(reversed(argspec.args), reversed(argspec.defaults)))


    def _get_child_inputs(self):
        return []
