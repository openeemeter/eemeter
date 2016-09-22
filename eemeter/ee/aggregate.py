from eemeter.ee.derivatives import DerivativePair, Derivative


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

    def __init__(self, aggregation_function="SUM",
                 baseline_default_value=None,
                 reporting_default_value=None):
        self.func = self.aggregation_functions[aggregation_function]
        self.baseline_default_value = baseline_default_value
        self.reporting_default_value = reporting_default_value

    def _validate_interpretation(self, derivative_pairs,
                                 target_interpretation):
        for d in derivative_pairs:
            if d.interpretation != target_interpretation:
                message = (
                    "DerivativePair interpretation ({}) does not match"
                    " target_interpretation ({})."
                    .format(d.interpretation, target_interpretation)
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

    def _get_valid_derivatives(self, derivative_pairs):
        baseline_derivatives, reporting_derivatives = [], []
        n_valid = 0
        n_invalid = 0

        for pair in derivative_pairs:

            baseline_derivative = pair.baseline
            baseline_valid = self._derivative_is_valid(baseline_derivative)
            if not baseline_valid:
                if self.baseline_default_value is not None:
                    baseline_derivative = self.baseline_default_value
                    baseline_valid = True

            reporting_derivative = pair.reporting
            reporting_valid = self._derivative_is_valid(reporting_derivative)
            if not reporting_valid:
                if self.reporting_default_value is not None:
                    reporting_derivative = self.reporting_default_value
                    reporting_valid = True

            if baseline_valid and reporting_valid:
                baseline_derivatives.append(baseline_derivative)
                reporting_derivatives.append(reporting_derivative)
                n_valid += 1
            else:
                n_invalid += 1

        return baseline_derivatives, reporting_derivatives, n_valid, n_invalid

    def _aggregate(self, derivatives):
        if len(derivatives) == 0:
            return None

        aggregated = derivatives[0]
        for d in derivatives[1:]:
            aggregated = self.func(aggregated, d)
        return aggregated

    def aggregate(self, derivative_pairs, target_interpretation=None,
                  target_unit=None):
        ''' Aggregates derivative pairs

        Parameters
        ----------
        derivative_pairs : list of eemeter.ee.meter.Derivative
            Derivative pairs to be aggregated; should be at least one.
        target_interpretation : str, default None
            Interpretation of derivatives; if None, will use interpretation of
            first.
        target_unit : str, default None
            Unit of derivatives; if None, will use unit of first.
        '''

        if len(derivative_pairs) == 0:
            raise ValueError(
                "Cannot aggregate empty list."
            )

        if target_interpretation is None:
            target_interpretation = derivative_pairs[0].interpretation

        if target_unit is None:
            target_unit = derivative_pairs[0].unit

        self._validate_interpretation(derivative_pairs, target_interpretation)
        self._validate_unit(derivative_pairs, target_unit)

        baseline_derivatives, reporting_derivatives, n_valid, n_invalid = \
            self._get_valid_derivatives(derivative_pairs)

        baseline_aggregation = self._aggregate(baseline_derivatives)
        reporting_aggregation = self._aggregate(reporting_derivatives)

        aggregated = DerivativePair(
            target_interpretation, target_unit,
            baseline_aggregation, reporting_aggregation
        )

        return aggregated, n_valid, n_invalid
