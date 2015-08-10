from .base import MeterBase

class And(MeterBase):
    """Performs an AND operation on input parameters and returns the result.

    Parameters
    ----------
    inputs : array_like
        Must contain the names of boolean parameters which must all be True
        in order for the output to be true.
    """
    def __init__(self, inputs, **kwargs):
        super(And, self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_raw(self, **kwargs):
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
    def __init__(self, inputs, **kwargs):
        super(Or, self).__init__(**kwargs)
        if len(inputs) == 0:
            message = "requires at least one input."
            raise ValueError(message)
        self.inputs = inputs

    def evaluate_raw(self, **kwargs):
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
