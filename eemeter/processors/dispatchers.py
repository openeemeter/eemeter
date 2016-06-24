import pandas as pd
import numpy as np

from eemeter.modeling.models.seasonal import SeasonalElasticNetCVModel
from eemeter.modeling.formatters import ModelDataFormatter
from eemeter.structures import EnergyTrace


default_formatter_settings = {
    'freq_str': 'D',
}


default_seasonal_settings = {
    'cooling_base_temp': 65,
    'heating_base_temp': 65,
}


ENERGY_MODEL_CLASS_MAPPING = {
    ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'H'): (
        ModelDataFormatter,
        default_formatter_settings,
        SeasonalElasticNetCVModel,
        default_seasonal_settings,
    ),
    ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'H'): (
        ModelDataFormatter,
        default_formatter_settings,
        SeasonalElasticNetCVModel,
        default_seasonal_settings,
    ),
    ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'H'): (
        ModelDataFormatter,
        default_formatter_settings,
        SeasonalElasticNetCVModel,
        default_seasonal_settings,
    ),
    ('NATURAL_GAS_CONSUMPTION_SUPPLIED', 'D'): (
        ModelDataFormatter,
        default_formatter_settings,
        SeasonalElasticNetCVModel,
        default_seasonal_settings,
    ),
    ('ELECTRICITY_CONSUMPTION_SUPPLIED', 'D'): (
        ModelDataFormatter,
        default_formatter_settings,
        SeasonalElasticNetCVModel,
        default_seasonal_settings,
    ),
    ('ELECTRICITY_ON_SITE_GENERATION_UNCONSUMED', 'D'): (
        ModelDataFormatter,
        default_formatter_settings,
        SeasonalElasticNetCVModel,
        default_seasonal_settings,
    ),
}


def dispatch_energy_modelers(logger, modeling_period_set, trace_set):

    for period_label, modeling_period in \
            modeling_period_set.get_modeling_periods():
        for trace_label, trace in trace_set.itertraces():

            energy_modeler = _get_energy_modeler(
                logger, period_label, modeling_period, trace, trace_label)

            yield (energy_modeler, period_label, trace_label)


def _get_energy_modeler(logger, period_label, modeling_period, trace,
                        trace_label):
    if trace.placeholder:
        logger.info("Skipping modeling for placeholder trace")
        return None

    filtered_data = _filter_by_modeling_period(trace, modeling_period)
    frequency = _get_approximate_frequency(logger, filtered_data, trace_label)
    model_class_selector = (trace.interpretation, frequency)

    try:
        formatter_class, formatter_settings, model_class, model_settings = \
            ENERGY_MODEL_CLASS_MAPPING[model_class_selector]
    except KeyError:
        message = (
            'Could not dispatch EnergyModeler/Model for'
            ' ModelingPeriod "{}" ({}) and trace "{}" ({}).'
            .format(period_label, modeling_period, trace_label,
                    trace.interpretation)
        )
        logger.error(message)
        return None

    formatter = formatter_class(**formatter_settings)
    model = model_class(**model_settings)
    filtered_trace = EnergyTrace(trace.interpretation, data=filtered_data,
                                 unit=trace.unit)
    return formatter, model, modeling_period, filtered_trace


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
    try:
        freq = pd.infer_freq(data.index)
    except ValueError:  # too few data points
        logger.error("Could not determine freqency - too few points.")
        return None
    else:
        if freq is not None:
            logger.info(
                "Determined frequency of {} for {}."
                .format(freq, trace_label)
            )
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
                return freq

    return None
