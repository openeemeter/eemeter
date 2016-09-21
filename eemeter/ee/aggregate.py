from eemeter.ee.derivatives import DerivativePair, Derivative


def sum_func(d1, d2):
    return Derivative(
        None,
        d1.value + d2.value,
        (d1.lower**2 + d2.lower**2)**0.5,
        (d1.upper**2 + d2.upper**2)**0.5,
        d1.n + d2.n,
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

    def _get_valid_derivatives(self, derivative_pairs):
        baseline_derivatives, reporting_derivatives = [], []
        n_valid = 0
        n_invalid = 0

        for pair in derivative_pairs:
            baseline_derivative = pair.baseline
            if baseline_derivative is None:
                if self.baseline_default_value is not None:
                    baseline_derivative = self.baseline_default_value

            reporting_derivative = pair.reporting
            if reporting_derivative is None:
                if self.reporting_default_value is not None:
                    reporting_derivative = self.reporting_default_value

            if baseline_derivative is not None and \
                    reporting_derivative is not None:
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

    def aggregate(self, derivative_pairs, target_interpretation):
        ''' Aggregates derivative pairs

        Parameters
        ----------
        derivative_pairs : list of eemeter.ee.meter.Derivative
            Derivative pairs to be aggregated.
        target_interpretation : str
            Interpretation of derivatives.
        '''
        self._validate_interpretation(derivative_pairs, target_interpretation)

        baseline_derivatives, reporting_derivatives, n_valid, n_invalid = \
            self._get_valid_derivatives(derivative_pairs)

        baseline_aggregation = self._aggregate(baseline_derivatives)
        reporting_aggregation = self._aggregate(reporting_derivatives)

        aggregated = DerivativePair(
            target_interpretation, baseline_aggregation, reporting_aggregation
        )

        return aggregated, n_valid, n_invalid
