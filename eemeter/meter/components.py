from .base import MeterBase

from itertools import chain
from pprint import pprint

class Sequence(MeterBase):
    def __init__(self,sequence,**kwargs):
        super(Sequence,self).__init__(**kwargs)
        assert all([issubclass(meter.__class__,MeterBase)
                    for meter in sequence])
        self.sequence = sequence

    def evaluate_mapped_inputs(self,**kwargs):
        """Collects and returns a series of meter object evaluation outputs in
        sequence, making the outputs of each meter available to those that
        follow.
        """
        result = {}
        for meter in self.sequence:
            args = {k:v for k,v in chain(kwargs.items(),result.items())}
            meter_result = meter.evaluate(**args)
            for k,v in meter_result.items():
                if k in result:
                    message = "unexpected repeated metric ({}) in {}. " \
                              "A different input_mapping or " \
                              "output_mapping may fix this overlap.".format(k,meter)
                    raise ValueError(message)
                result[k] = v
        return result

    def _get_child_inputs(self):
        inputs = []
        for meter in self.sequence:
            inputs.append(meter.get_inputs())
        return inputs

class Condition(MeterBase):
    def __init__(self,condition_parameter,success=None,failure=None,**kwargs):
        super(Condition,self).__init__(**kwargs)
        self.condition_parameter = condition_parameter
        self.success = success
        self.failure = failure

    def evaluate_mapped_inputs(self,**kwargs):
        """Returns evaluations for either the `success` meter or the `failure`
        meter depending on the boolean value of the meter input or output with
        the name stored in `condition_parameter`.
        """
        if kwargs[self.condition_parameter]:
            if self.success is None:
                return {}
            else:
                return self.success.evaluate(**kwargs)
        else:
            if self.failure is None:
                return {}
            else:
                return self.failure.evaluate(**kwargs)

    def _get_child_inputs(self):
        inputs = {}
        if self.success is not None:
            inputs["success"] = self.success.get_inputs()
        if self.failure is not None:
            inputs["failure"] = self.success.get_inputs()
        return inputs

class And(MeterBase):
    def __init__(self,inputs,**kwargs):
        super(And,self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_mapped_inputs(self,**kwargs):
        output = True
        for inpt in self.inputs:
            boolean = kwargs.get(inpt)
            if boolean is None:
                message = "could not find input '{}'".format(inpt)
                raise ValueError(message)
            output = output and boolean
        return {"output": output}

class Or(MeterBase):
    def __init__(self,inputs,**kwargs):
        super(Or,self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_mapped_inputs(self,**kwargs):
        output = False
        for inpt in self.inputs:
            boolean = kwargs.get(inpt)
            if boolean is None:
                message = "could not find input '{}'".format(inpt)
                raise ValueError(message)
            output = output or boolean
        return {"output": output}

class Switch(MeterBase):
    def __init__(self,target,cases,default=None,**kwargs):
        super(Switch,self).__init__(**kwargs)
        self.target = target
        self.cases = cases
        self.default = default

    def evaluate_mapped_inputs(self,**kwargs):
        item = kwargs.get(self.target)
        if item is None:
            return {}
        meter = self.cases.get(item)
        if meter is not None:
            return meter.evaluate(**kwargs)
        if self.default is not None:
            return self.default.evaluate(**kwargs)
        return {}
