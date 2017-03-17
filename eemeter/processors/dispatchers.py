import logging

import pandas as pd

from eemeter.modeling.models.seasonal import SeasonalElasticNetCVModel
from eemeter.modeling.models.billing import BillingElasticNetCVModel
from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter,
)
from eemeter.modeling.split import (
    SplitModeledEnergyTrace
)

logger = logging.getLogger(__name__)

default_dispatch = (
    ModelDataFormatter,
    {
        'freq_str': 'D'
    },
    SeasonalElasticNetCVModel,
    {
        'cooling_base_temp': 65,
        'heating_base_temp': 65,
    },
)


billing_dispatch = (
    ModelDataBillingFormatter,
    {},
    BillingElasticNetCVModel,
    {
        'cooling_base_temp': 65,
        'heating_base_temp': 65,
    },
)


ENERGY_MODEL_CLASS_MAPPING = {
    ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '15T'): default_dispatch,
    ('ELECTRICITY_CONSUMPTION_SUPPLIED', '15T'): default_dispatch,
    ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '15T'): default_dispatch,

    ('NATURAL_GAS_CONSUMPTION_SUPPLIED', '30T'): default_dispatch,
    ('ELECTRICITY_CONSUMPTION_SUPPLIED', '30T'): default_dispatch,
    ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', '30T'): default_dispatch,

    ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'): default_dispatch,
    ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'): default_dispatch,
    ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'): default_dispatch,

    ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'): default_dispatch,
    ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'): default_dispatch,
    ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'): default_dispatch,

    ('NATURAL_GAS_CONSUMPTION_SUPPLIED', None): billing_dispatch,
    ('ELECTRICITY_CONSUMPTION_SUPPLIED', None): billing_dispatch,
}


def get_energy_modeling_dispatches(modeling_period_set, trace_set):
    ''' Dispatches a set of applicable models and formatters for each
    pairing of modeling period sets and trace sets given.

    Parameters
    ----------
    modeling_period_set : eemeter.structures.ModelingPeriodSet

        :code:`ModelingPeriod` s to dispatch.
    trace_set : eemeter.structures.EnergyTraceSet
        :code:`EnergyTrace` s to dispatch.
    '''

    dispatches = {}
    for trace_label, trace in trace_set.itertraces():

        dispatches[trace_label] = None

        if trace.placeholder:
            logger.debug(
                'Skipping modeling for placeholder trace "{}" ({}).'
                .format(trace_label, trace.interpretation)
            )
            continue

        frequency = get_approximate_frequency(trace)

        if frequency not in ['H', 'D', '15T', '30T']:
            frequency = None

        model_class_selector = (trace.interpretation, frequency)

        try:
            (
                FormatterClass,
                formatter_settings,
                ModelClass,
                model_settings,
            ) = ENERGY_MODEL_CLASS_MAPPING[model_class_selector]
        except KeyError:
            logger.error(
                'Could not dispatch formatter/model for'
                ' model class selector {}.'
                .format(model_class_selector)
            )
            continue

        formatter = FormatterClass(**formatter_settings)
        model = ModelClass(**model_settings)

        model_mapping = {
            modeling_period_label: ModelClass(**model_settings)
            for modeling_period_label, _ in
            modeling_period_set.iter_modeling_periods()
        }

        modeled_energy_trace = SplitModeledEnergyTrace(
            trace, formatter, model_mapping, modeling_period_set)

        logger.debug(
            'Successfully created SplitModeledEnergyTrace formatter {}'
            ' and model {} for {} and trace "{}" ({})'
            ' using model class selector {}.'
            .format(formatter, model, modeling_period_set,
                    trace_label, trace.interpretation, model_class_selector)
        )

        dispatches[trace_label] = modeled_energy_trace

    return dispatches


def get_approximate_frequency(trace):

    if trace.data is None:
        logger.warn(
            "Could not determine frequency:"
            " {} is placeholder instance."
            .format(trace)
        )
        return None

    def _log_success(freq):
        logger.debug(
            "Determined frequency of '{}' for {}."
            .format(freq, trace)
        )

    try:
        freq = pd.infer_freq(trace.data.index)
    except ValueError:  # too few data points
        logger.error("Could not determine frequency - too few points.")
        return None
    else:
        if freq is not None:
            _log_success(freq)
            return freq

    # freq is None - maybe because of a DST change (23/25 hours)?
    # strategy: try two groups of 5 dates
    for i in range(0, 9, 5):
        try:
            freq = pd.infer_freq(trace.data.index[i:i + 5])
        except ValueError:
            pass
        else:
            if freq is not None:
                _log_success(freq)
                return freq

    logger.warning("Could not determine frequency - no dominant frequency.")
    return None
