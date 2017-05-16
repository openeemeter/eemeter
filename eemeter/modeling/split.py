import logging
import traceback

import numpy as np

import eemeter.modeling.exceptions as model_exceptions
from eemeter.structures import EnergyTrace

logger = logging.getLogger(__name__)


class SplitModeledEnergyTrace(object):
    ''' Light wrapper around models applicable to a single trace which
    fits and predicts multiple models for different segments.

    Parameters
    ----------
    trace : eemeter.structures.EnergyTrace
        Trace to be modeled.
    formatter : eemeter.modeling.formatter.Formatter
        Formatter to prep trace data for modeling.
    model_mapping : dict
        Items of this dictionary map `modeling_period_label` s to models
    modeling_period_set : eemeter.structures.ModelingPeriodSet
        The set of modeling periods over which models should be applicable.
    '''

    def __init__(self, trace, formatter, model_mapping, modeling_period_set):
        self.trace = trace
        self.formatter = formatter
        self.model_mapping = model_mapping
        self.modeling_period_set = modeling_period_set
        self.fit_outputs = {}

    def __repr__(self):
        return (
            "SplitModeledEnergyTrace(trace={}, formatter={},"
            " model_mapping={}, modeling_period_set={})"
            .format(self.trace, self.formatter, self.model_mapping,
                    self.modeling_period_set)
        )

    def fit(self, weather_source):
        ''' Fit all models associated with this trace.

        Parameters
        ----------
        weather_source : eemeter.weather.ISDWeatherSource
            Weather source to use in creating covariate data.
        '''

        for modeling_period_label, modeling_period in \
                self.modeling_period_set.iter_modeling_periods():

            filtered_data = self._filter_by_modeling_period(
                self.trace, modeling_period)
            filtered_trace = EnergyTrace(
                self.trace.interpretation, data=filtered_data,
                unit=self.trace.unit)

            model = self.model_mapping[modeling_period_label]

            outputs = {
                "status": None,
                "traceback": None,
                "input_data": None,
                "start_date": None,
                "end_date": None,
                "n_rows": None,
                "model_fit": {},
            }

            # fail with DataSufficiencyException if bad weather source
            if weather_source is None:
                message = (
                    'No weather source found for trace {} in {} period'
                    .format(self.trace.trace_id, modeling_period_label)
                )
                logger.warn(message)
                try:
                    raise model_exceptions.DataSufficiencyException(message)
                except:
                    outputs.update({
                        "status": "FAILURE",
                        "traceback": traceback.format_exc(),
                    })
                    self.fit_outputs[modeling_period_label] = outputs
                continue

            # attempt to create input data
            try:
                input_data = self.formatter.create_input(
                    filtered_trace, weather_source)
            except:
                logger.warn(
                    'Input data formatting failed for trace {} in {} period.'
                    .format(self.trace.trace_id, modeling_period_label)
                )
                outputs.update({
                    "status": "FAILURE",
                    "traceback": traceback.format_exc(),
                })
            else:
                input_description = self.formatter.describe_input(input_data)
                input_serialization = self.formatter.serialize_input(
                    input_data)
                input_mask = self.formatter.get_input_data_mask(
                    input_data)
                outputs.update({
                    "input_data_serialization": input_serialization,
                    "input_mask": input_mask,  # missing days
                    "start_date": input_description.get('start_date'),
                    "end_date": input_description.get('end_date'),
                    "n_rows": input_description.get('n_rows'),
                    "trace": filtered_trace,
                })

                try:
                    model_fit = model.fit(input_data)
                except:
                    tb = traceback.format_exc()
                    logger.warn(
                        '{} fit failed for trace {} in {} period.'
                        .format(model, self.trace.trace_id, modeling_period_label)
                    )

                    outputs.update({
                        "status": "FAILURE",
                        "traceback": tb,
                    })
                else:
                    logger.debug(
                        '{} fit successful for trace {} in {} period.'
                        .format(model, self.trace.trace_id, modeling_period_label)
                    )
                    outputs["model_fit"].update(model_fit)
                    outputs.update({
                        "status": "SUCCESS",
                    })

            self.fit_outputs[modeling_period_label] = outputs

        return self.fit_outputs

    def predict(self, modeling_period_label, demand_fixture_data, **kwargs):
        ''' Predict for any one of the modeling_periods associated with this
        trace. Light wrapper around :code:`model.predict(` method.

        Parameters
        ----------
        modeling_period_label : str
            Modeling period indicating which model to use in making the
            prediction.
        demand_fixture_data : object
            Data (formatted by :code:`self.formatter`) over which prediction
            should be made.
        params : object, default None
            Fitted parameters for the model. If :code:`None`, use parameters
            found when :code:`.fit(` method was called.
        '''

        outputs = self.fit_outputs[modeling_period_label]
        model = self.model_mapping[modeling_period_label]

        if outputs["status"] == "FAILURE":
            logger.warn(
                '{} cannot predict in failed {} period for trace {}.'
                .format(model, modeling_period_label, self.trace.trace_id)
            )
            return None

        if 'params' not in kwargs:
            kwargs['params'] = outputs["model_fit"]["model_params"]

        return model.predict(demand_fixture_data, **kwargs)

    def compute_derivative(self, modeling_period_label, derivative_callable,
                           derivative_callable_kwargs):
        ''' Compute a modeling derivative for this modeling period.

        Parameters
        ----------
        modeling_period_label : str
            Label for modeling period for which derivative should be computed.
        derivative_callable : callable
            Callable which can be used as follows:

            .. code-block: python

                >>> derivative_callable(formatter, model, **kwargs)

        derivative_callable_kwargs : dict
            Arbitrary keyword arguments to be passed to the derviative callable
        '''

        outputs = self.fit_outputs[modeling_period_label]

        if outputs["status"] == "FAILURE":
            return None

        model = self.model_mapping[modeling_period_label]

        try:
            derivative = derivative_callable(self.formatter, model,
                                             **derivative_callable_kwargs)
        except Exception:
            logger.exception("Derivative computation failed.")
            return None

        return derivative

    @staticmethod
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
            last_index = filtered_df.index[-1]
            filtered_df.set_value(last_index, 'value', np.nan)
            filtered_df.set_value(last_index, 'estimated', False)

        return filtered_df
