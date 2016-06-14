from eemeter.modeling.models.seasonal import SeasonalElasticNetCVModel
from eemeter.modeling.modelers import DailyModeler
import pandas as pd

default_seasonal_settings = {
    "cooling_base_temp": 65,
    "heating_base_temp": 65,
}


class EnergyModelerDispatcher(object):

    ENERGY_MODEL_CLASS_MAPPING = {
        ('natural_gas', 'CONSUMPTION_SUPPLIED', 'H'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
        ('electricity', 'CONSUMPTION_SUPPLIED', 'H'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
        ('natural_gas', 'ON_SITE_GENERATION_UNCONSUMED', 'H'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
        ('electricity', 'ON_SITE_GENERATION_UNCONSUMED', 'H'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
        ('natural_gas', 'CONSUMPTION_SUPPLIED', 'D'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
        ('electricity', 'CONSUMPTION_SUPPLIED', 'D'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
        ('natural_gas', 'ON_SITE_GENERATION_UNCONSUMED', 'D'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
        ('electricity', 'ON_SITE_GENERATION_UNCONSUMED', 'D'): (
            DailyModeler,
            SeasonalElasticNetCVModel,
            default_seasonal_settings,
        ),
    }

    def dispatch_energy_modelers(self, modeling_period_set, trace_set):

        for period_label, modeling_period in \
                modeling_period_set.get_modeling_periods():
            for trace_label, trace in trace_set.get_traces():

                energy_modeler = self._get_energy_modeler(
                        modeling_period, trace)

                validation_errors = []

                # skip if couldn't find a good model, also note in errors:
                if energy_modeler is None:
                    message = (
                        'Could not dispatch EnergyModeler/Model for'
                        ' ModelingPeriod "{}" ({}) and trace "{}" ({}, {}).'
                        .format(period_label, modeling_period, trace_label,
                                trace.fuel, trace.interpretation)
                    )
                    validation_errors.append(message)

                results = (energy_modeler, period_label, trace_label)
                yield results, validation_errors

    def _get_energy_modeler(self, modeling_period, trace):
        filtered_trace = trace.filter_by_modeling_period(modeling_period)
        frequency = self._get_approximate_frequency(filtered_trace)
        model_class_selector = (trace.fuel, trace.interpretation, frequency)

        try:
            modeler_class, model_class, model_settings = \
                self.ENERGY_MODEL_CLASS_MAPPING[model_class_selector]
        except KeyError:
            return None

        model = model_class(**model_settings)
        return modeler_class(model, modeling_period, filtered_trace)

    def _get_approximate_frequency(self, trace):
        try:
            freq = pd.infer_freq(trace.data.index)
        except ValueError:  # too few data points
            return None
        else:
            if freq is not None:
                return freq

        # freq is None - maybe because of a DST change (23/25 hours)?
        # strategy: try two groups of 5 dates
        for i in range(0, 9, 5):
            try:
                freq = pd.infer_freq(trace.data.index[i:i+5])
            except ValueError:
                pass
            else:
                if freq is not None:
                    return freq

        return None
