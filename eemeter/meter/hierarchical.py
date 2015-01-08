class MeterBase(object):
    def __init__(self,**kwargs):
        if "input_mapping" in kwargs:
            self.input_mapping = kwargs["input_mapping"]
        else:
            self.input_mapping = {}
        if "output_mapping" in kwargs:
            self.output_mapping = kwargs["output_mapping"]
        else:
            self.output_mapping = {}

    def evaluate(self,**kwargs):
        mapped_inputs = self.apply_input_mapping(kwargs)
        result = self.evaluate_mapped_inputs(**mapped_inputs)
        mapped_output = self.apply_output_mapping(result)
        return mapped_output

    def apply_input_mapping(self,inputs):
        mapped_inputs = {}
        for k,v in inputs.iteritems():
            if k in self.input_mapping:
                mapped_inputs[self.input_mapping[k]] = v
            else:
                mapped_inputs[k] = v
        return mapped_inputs

    def apply_output_mapping(self,outputs):
        mapped_outputs = {}
        for k,v in outputs.iteritems():
            if k in self.output_mapping:
                mapped_outputs[self.output_mapping[k]] = v
            else:
                mapped_outputs[k] = v
        return mapped_outputs

    def evaluate_mapped_inputs(self,**kwargs):
        raise NotImplementedError

class SequentialMeter(MeterBase):
    def __init__(self,sequence,**kwargs):
        super(SequentialMeter,self).__init__(**kwargs)
        assert all([issubclass(meter.__class__,MeterBase)
                    for meter in sequence])
        self.sequence = sequence

    def evaluate_mapped_inputs(self,**kwargs):
        result = {}
        for meter in self.sequence:
            meter_result = meter.evaluate(**kwargs)
            for k,v in meter_result.iteritems():
                if k in result:
                    message = "unexpected repeated metric ({}). " \
                              "A different input_mapping or " \
                              "output_mapping may fix this overlap."
                    raise ValueError(message.format(k))
                result[k] = v
        return result

class DummyMeter(MeterBase):
    def evaluate_mapped_inputs(self,**kwargs):
        result = {"value": kwargs["value"]}
        return result
