import inspect

class MeterBase(object):
    """Base class for all Meter objects. Takes care of structural tasks such as
    input and output mapping.
    """
    def __init__(self,input_mapping={},output_mapping={},extras={},**kwargs):
        """- `input_mapping` should be a dictionary with incoming input names
             as keys whose associated values are outgoing input names.
             (e.g. ({"old_input_name":"new_input_name"})
           - `output_mapping` should be a dictionary with incoming output names
             as keys whose associated values are outgoing output names.
             (e.g. ({"old_output_name":"new_output_name"})
        """
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.extras = extras

    def evaluate(self,**kwargs):
        """Returns a dictionary of evaluated meter outputs. Arguments must be
        specified as keyword arguments. **Note:** because `**kwargs` is
        intentionally general (and therefore woefully unspecific), the function
        `meter.get_inputs()` exists to help describe the required inputs.
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
        for inpt in inputs:
            if inpt not in mapped_inputs:
                message = "expected argument '{}' for meter '{}'; "\
                          "got kwargs={} (with mapped_inputs={}) instead."\
                                  .format(inpt,self.__class__.__name__,
                                          sorted(kwargs.items()),sorted(mapped_inputs.items()))
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
        meter, returning a dictionary of meter outputs.
        """
        raise NotImplementedError

    def get_inputs(self):
        """Returns a recursively structured object which shows necessary
        inputs.
        """
        inputs = inspect.getargspec(self.evaluate_mapped_inputs).args[1:]
        child_inputs = self._get_child_inputs()
        return {self.__class__.__name__:{"inputs":inputs,"child_inputs":child_inputs}}

    def _get_child_inputs(self):
        return []

