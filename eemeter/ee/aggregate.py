from collections import OrderedDict

from eemeter.ee.derivatives import DerivativePair, Derivative
from eemeter.io.serializers import (
    deserialize_aggregation_input,
    serialize_derivative_pair,
)


def sum_func(d1, d2):
    return Derivative(
        None,
        d1.value + d2.value,
        (d1.lower**2 + d2.lower**2)**0.5,
        (d1.upper**2 + d2.upper**2)**0.5,
        d1.n + d2.n,
        None,
    )


class Aggregator(object):
    """
    Enforces trace interpretation uniformity, aggregates according to the
    aggregation rules supplied.
    """

    aggregation_functions = {
        "SUM": sum_func,
    }

    def _validate_derivative_interpretation(self, derivative_pairs,
                                            target_derivative_interpretation):
        for d in derivative_pairs:
            if d.derivative_interpretation != target_derivative_interpretation:
                message = (
                    "DerivativePair derivative_interpretation ({}) does not"
                    " match target_derivative_interpretation ({})."
                    .format(d.derivative_interpretation,
                            target_derivative_interpretation)
                )
                raise ValueError(message)

    def _validate_trace_interpretation(self, derivative_pairs,
                                       target_trace_interpretation):
        for d in derivative_pairs:
            if d.trace_interpretation != target_trace_interpretation:
                message = (
                    "DerivativePair trace_interpretation ({}) does not"
                    " match target_trace_interpretation ({})."
                    .format(d.trace_interpretation,
                            target_trace_interpretation)
                )
                raise ValueError(message)

    def _validate_unit(self, derivative_pairs, target_unit):
        for d in derivative_pairs:
            if d.unit != target_unit:
                message = (
                    "DerivativePair unit ({}) does not match"
                    " target_unit ({})."
                    .format(d.unit, target_unit)
                )
                raise ValueError(message)

    def _derivative_is_valid(self, derivative):
        return not (
            derivative.value is None or
            derivative.lower is None or
            derivative.upper is None or
            derivative.n is None
        )

    def _get_valid_derivatives(self, derivative_pairs, baseline_default_value,
                               reporting_default_value):
        baseline_derivatives, reporting_derivatives = [], []
        statuses = OrderedDict([])

        for pair in derivative_pairs:

            baseline_status = "ACCEPTED"
            baseline_derivative = pair.baseline
            baseline_valid = self._derivative_is_valid(baseline_derivative)
            if not baseline_valid:
                baseline_status = "REJECTED"
                if baseline_default_value is not None:
                    baseline_status = "DEFAULT"
                    baseline_derivative = baseline_default_value

            reporting_status = "ACCEPTED"
            reporting_derivative = pair.reporting
            reporting_valid = self._derivative_is_valid(reporting_derivative)
            if not reporting_valid:
                reporting_status = "REJECTED"
                if reporting_default_value is not None:
                    reporting_status = "DEFAULT"
                    reporting_derivative = reporting_default_value

            if ((baseline_status in ["ACCEPTED", "DEFAULT"]) and
                    (reporting_status in ["ACCEPTED", "DEFAULT"])):
                baseline_derivatives.append(baseline_derivative)
                reporting_derivatives.append(reporting_derivative)
                pair_status = "ACCEPTED"
            else:
                pair_status = "REJECTED"

            statuses[pair.label] = OrderedDict([
                ("baseline_status", baseline_status),
                ("reporting_status", reporting_status),
                ("status", pair_status),
            ])

        return baseline_derivatives, reporting_derivatives, statuses

    def _aggregate(self, derivatives, func):
        if len(derivatives) == 0:
            return None

        aggregated = derivatives[0]
        for d in derivatives[1:]:
            aggregated = func(aggregated, d)
        return aggregated

    def aggregate(self, aggregation_input):
        ''' Aggregates derivative pairs

        Parameters
        ----------
        aggregation_input : str, default None
            Unit of derivatives; if None, will use unit of first.
        '''
        deserialized = deserialize_aggregation_input(aggregation_input)

        aggregation_interpretation = deserialized["aggregation_interpretation"]
        func = self.aggregation_functions[aggregation_interpretation]
        baseline_default_value = deserialized.get("baseline_default_value")
        reporting_default_value = deserialized.get("reporting_default_value")

        derivative_pairs = deserialized["derivative_pairs"]
        target_derivative_interpretation = \
            deserialized["derivative_interpretation"]
        target_trace_interpretation = deserialized["trace_interpretation"]
        target_unit = None

        if len(derivative_pairs) == 0:
            raise ValueError(
                "Cannot aggregate empty list."
            )

        if target_derivative_interpretation is None:
            target_derivative_interpretation = \
                derivative_pairs[0].derivative_interpretation

        if target_trace_interpretation is None:
            target_trace_interpretation = \
                derivative_pairs[0].trace_interpretation

        if target_unit is None:
            target_unit = derivative_pairs[0].unit

        self._validate_derivative_interpretation(
            derivative_pairs, target_derivative_interpretation)
        self._validate_trace_interpretation(
            derivative_pairs, target_trace_interpretation)
        self._validate_unit(derivative_pairs, target_unit)

        baseline_derivatives, reporting_derivatives, status = \
            self._get_valid_derivatives(derivative_pairs,
                                        baseline_default_value,
                                        reporting_default_value)

        baseline_aggregation = self._aggregate(baseline_derivatives, func)
        reporting_aggregation = self._aggregate(reporting_derivatives, func)

        if baseline_aggregation is None or reporting_aggregation is None:
            aggregated = None
        else:
            aggregated = DerivativePair(
                None, target_derivative_interpretation,
                target_trace_interpretation, target_unit,
                baseline_aggregation, reporting_aggregation
            )
            aggregated = serialize_derivative_pair(aggregated)

        return OrderedDict([
            ("aggregated", aggregated),
            ("aggregation_interpretation", aggregation_interpretation),
            ("derivative_interpretation", target_derivative_interpretation),
            ("trace_interpretation", target_trace_interpretation),
            ("unit", target_unit),
            ("status", status),
        ])
