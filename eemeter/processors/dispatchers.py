import pandas as pd
import numpy as np

from eemeter.modeling.models.seasonal import SeasonalElasticNetCVModel
from eemeter.modeling.models.billing import BillingElasticNetCVModel
from eemeter.modeling.formatters import (
    ModelDataFormatter,
    ModelDataBillingFormatter,
)
from eemeter.structures import EnergyTrace


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


def get_energy_modeling_dispatches(logger, modeling_period_set, trace_set):
    '''
    '''

    dispatches = {}
    for mp_label, modeling_period in \
            modeling_period_set.get_modeling_periods():
        for t_label, trace in trace_set.itertraces():

            frequency = _get_approximate_frequency(
                    logger, trace.data, t_label)

            if frequency not in ['H', 'D', '15T', '30T']:
                frequency = None

            model_class_selector = (trace.interpretation, frequency)

            formatter, model, filtered_trace = _dispatch(
                logger, model_class_selector, modeling_period, trace,
                mp_label, t_label)

            dispatches[(mp_label, t_label)] = {
                "formatter": formatter,
                "model": model,
                "filtered_trace": filtered_trace,
            }

    return dispatches


def _dispatch(logger, model_class_selector, modeling_period, trace,
              modeling_period_label, trace_label):

    if trace.placeholder:
        logger.info(
            'Skipping modeling for placeholder trace "{}" ({}).'
            .format(trace_label, trace.interpretation)
        )
        return (None, None, None)

    try:
        formatter_class, formatter_settings, model_class, model_settings = \
            ENERGY_MODEL_CLASS_MAPPING[model_class_selector]
    except KeyError:
        logger.error(
            'Could not dispatch formatter/model for'
            ' ModelingPeriod "{}" ({}) and trace "{}" ({}) using model class'
            ' selector {}.'
            .format(modeling_period_label, modeling_period, trace_label,
                    trace.interpretation, model_class_selector)
        )
        return (None, None, None)

    formatter = formatter_class(**formatter_settings)
    model = model_class(**model_settings)

    filtered_data = _filter_by_modeling_period(trace, modeling_period)
    filtered_trace = EnergyTrace(trace.interpretation, data=filtered_data,
                                 unit=trace.unit)
    logger.info(
        'Successfully dispatched formatter {} and model {} for'
        ' ModelingPeriod "{}" ({}) and trace "{}" ({}) using model class'
        ' selector {}.'
        .format(formatter, model, modeling_period_label, modeling_period,
                trace_label, trace.interpretation, model_class_selector)
    )
    return formatter, model, filtered_trace


def _filter_by_modeling_period(trace, modeling_period):

    start = modeling_period.start_date
    end = modeling_period.end_date

    if start is None:
        if end is None:
            filtered_df = trace.data.copy()
        else:
            filtered_df = trace.data[:end].copy()
    else:
        if end is None:
            filtered_df = trace.data[start:].copy()
        else:
            filtered_df = trace.data[start:end].copy()

    # require NaN last data point as cap
    if filtered_df.shape[0] > 0:
        filtered_df.value.iloc[-1] = np.nan
        filtered_df.estimated.iloc[-1] = False

    return filtered_df


def _get_approximate_frequency(logger, data, trace_label):

    if data is None:
        logger.info(
            "Could not determine frequency:"
            " EnergyTrace '{}' is placeholder instance."
            .format(trace_label)
        )
        return None

    def _log_success(freq):
        logger.info(
            "Determined frequency of '{}' for EnergyTrace '{}'."
            .format(freq, trace_label)
        )

    try:
        freq = pd.infer_freq(data.index)
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
            freq = pd.infer_freq(data.index[i:i+5])
        except ValueError:
            pass
        else:
            if freq is not None:
                _log_success(freq)
                return freq

    logger.warning("Could not determine frequency - no dominant frequency.")
    return None
