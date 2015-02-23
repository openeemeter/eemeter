from eemeter.meter import MeterBase
from eemeter.config.yaml_parser import load

class BPI2400Meter(MeterBase):
    """Implementation of BPI-2400 standard
    """

    def __init__(self,**kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.meter = load(self._meter_yaml())

    def _meter_yaml(self):
        meter_yaml = """
            !obj:eemeter.meter.Sequence {
                sequence: [
                    !obj:eemeter.meter.RecentReadingMeter {
                        n_days: 360
                    },
                    !obj:eemeter.meter.EstimatedReadingConsolidationMeter {
                    },
                    !obj:eemeter.meter.And {
                        inputs: [
                            recent_reading,
                        ],
                        output_mapping: {
                            output: meets_model_calibration_utility_bill_criteria
                        }
                    },
                    !obj:eemeter.meter.Condition {
                        condition_parameter: meets_model_calibration_utility_bill_criteria,
                        success: !obj:eemeter.meter.DummyMeter {
                            input_mapping: {
                                consumption_history_no_estimated: value
                            }
                        },
                        failure: !obj:eemeter.meter.DummyMeter {
                            input_mapping: {
                                consumption_history_no_estimated: value
                            }
                        }
                    }
                ]
            }
            """
        return meter_yaml

    def evaluate_mapped_inputs(self,**kwargs):
        return self.meter.evaluate(**kwargs)

    def _get_child_inputs(self):
        return self.meter.get_inputs()
