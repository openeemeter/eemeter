from .base import MeterBase

from itertools import chain

class Sequence(MeterBase):
    """Collects and returns a series of meter object evaluation outputs in
    sequence, making the outputs of each meter available to those that
    follow.

    Parameters
    ----------
    sequence : array_like
        Meters will be executed in the order they are given; meters coming
        later in sequence will see the original inputs and the additional
        collected outputs from previously executed meters.
    """

    def __init__(self,sequence,**kwargs):
        super(Sequence,self).__init__(**kwargs)
        assert all([issubclass(meter.__class__,MeterBase)
                    for meter in sequence])
        self.sequence = sequence

    def evaluate_mapped_inputs(self,**kwargs):
        """Evaluates meters in sequence, collecting outputs.

        Returns
        -------
        out : dict
            Collected outputs from all meters in the sequence.
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
    """Collects and returns a series of meter object evaluation outputs in
    sequence, making the outputs of each meter available to those that
    follow.

    Parameters
    ----------
    condition_parameter : str
        The name of the parameter which contains a boolean on which to
        condition.
    success : eemeter.meter.MeterBase, optional
        The meter to execute if the condition is True.
    failure : eemeter.meter.MeterBase, optional
        The meter to execute if the condition is False.
    """

    def __init__(self,condition_parameter,success=None,failure=None,**kwargs):
        super(Condition,self).__init__(**kwargs)
        self.condition_parameter = condition_parameter
        self.success = success
        self.failure = failure

    def evaluate_mapped_inputs(self,**kwargs):
        """Evaluate either the `success` meter or the `failure`
        meter depending on the boolean value of the meter input or output with
        the name stored in `condition_parameter`.

        Returns
        -------
        out : dict
            Collected outputs from either the success or the failure meter.
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
    """Performs an AND operation on input parameters and returns the result.

    Parameters
    ----------
    inputs : array_like
        Must contain the names of boolean parameters which must all be True
        in order for the output to be true.
    """
    def __init__(self,inputs,**kwargs):
        super(And,self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_mapped_inputs(self,**kwargs):
        """Collect the values of the given parameters and returns True if all
        are True.

        Returns
        -------
        out : dict
            A dictionary with a single key, 'output', which has the value of
            the boolean result.
        """
        output = True
        for inpt in self.inputs:
            boolean = kwargs.get(inpt)
            if boolean is None:
                message = "could not find input '{}'".format(inpt)
                raise ValueError(message)
            output = output and boolean
        return {"output": output}

class Or(MeterBase):
    """Performs an OR operation on input parameters and returns the result.

    Parameters
    ----------
    inputs : array_like
        Must contain the names of boolean parameters at least one of which must
        be True in order for the output to be true.
    """
    def __init__(self,inputs,**kwargs):
        super(Or,self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_mapped_inputs(self,**kwargs):
        """Collect the values of the given parameters and returns True if at
        least one is True.

        Returns
        -------
        out : dict
            A dictionary with a single key, 'output', which has the value of
            the boolean result.
        """
        output = False
        for inpt in self.inputs:
            boolean = kwargs.get(inpt)
            if boolean is None:
                message = "could not find input '{}'".format(inpt)
                raise ValueError(message)
            output = output or boolean
        return {"output": output}

class Switch(MeterBase):
    """Switches between cases of values of a particular parameter.

    Parameters
    ----------
    target : str
        The name of the parameter on which to switch.
    cases : dict
        A dictionary of meters to execute depending on the value of the target
        parameter; keyed on potential values of target.
    default : eemeter.meter.MeterBase, optional
        The meter to exectute if the value of the parameter was not found in
        the dictionary of cases.
    """
    def __init__(self,target,cases,default=None,**kwargs):
        super(Switch,self).__init__(**kwargs)
        self.target = target
        self.cases = cases
        self.default = default

    def evaluate_mapped_inputs(self,**kwargs):
        """Determine the value of the target parameter and, according to that
        value, choose which meter to run.

        Returns
        -------
        out : dict
            Contains the results of the meter which was run.
        """
        item = kwargs.get(self.target)
        if item is None:
            return {}
        meter = self.cases.get(item)
        if meter is not None:
            return meter.evaluate(**kwargs)
        if self.default is not None:
            return self.default.evaluate(**kwargs)
        return {}
