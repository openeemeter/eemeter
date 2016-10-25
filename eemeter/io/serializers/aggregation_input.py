from eemeter.ee.derivatives import Derivative, DerivativePair


def deserialize_aggregation_input(aggregation_input):

    # verify type
    type_ = aggregation_input.get('type', None)
    if type_ is None:
        return {
            'error': 'Serialization "type" must be provided for'
            ' aggregation_input.'
        }

    # switch on type
    if type_ == 'BASIC_AGGREGATION':
        return _deserialize_basic_aggregation(aggregation_input)
    else:
        return {
            'error': 'Serialization type "{}" not recognized.'.format(type_)
        }


def _get_key_or_error(serialized, key_name, type_):

    value = serialized.get(key_name, None)

    error = None
    if value is None:
        error = {
            'error': (
                'For serialization type "{}",'
                ' key "{}" must be provided.'
                .format(type_, key_name)
            )
        }

    return value, error


def _deserialize_basic_aggregation(aggregation_input):
    type_ = 'BASIC_AGGREGATION'

    # check for "aggregation_interpretation" key
    aggregation_interpretation, error = _get_key_or_error(
        aggregation_input, "aggregation_interpretation", type_)
    if aggregation_interpretation is None:
        return error

    # check for "trace_interpretation" key
    trace_interpretation, error = _get_key_or_error(
        aggregation_input, "trace_interpretation", type_)
    if trace_interpretation is None:
        return error

    # check for "derivative_interpretation" key
    derivative_interpretation, error = _get_key_or_error(
        aggregation_input, "derivative_interpretation", type_)
    if derivative_interpretation is None:
        return error

    # check for "derivatives" key
    derivatives, error = _get_key_or_error(
        aggregation_input, "derivatives", type_)
    if derivatives is None:
        return error

    # check for baseline default value
    baseline_default_value = aggregation_input.get("baseline_default_value")
    if baseline_default_value is None:
        baseline_default = None
    else:
        baseline_default = _deserialize_default_value(baseline_default_value)
        if "error" in baseline_default:
            return baseline_default

    # check for reporting default value
    reporting_default_value = aggregation_input.get("reporting_default_value")
    if reporting_default_value is None:
        reporting_default = None
    else:
        reporting_default = _deserialize_default_value(reporting_default_value)
        if "error" in reporting_default:
            return reporting_default

    # check for "derivatives" key
    derivatives, error = _get_key_or_error(
        aggregation_input, "derivatives", type_)
    if derivatives is None:
        return error

    derivative_pairs = _deserialize_derivatives(derivatives)
    if "error" in derivative_pairs:
        return derivative_pairs

    return {
        "aggregation_interpretation": aggregation_interpretation,
        "trace_interpretation": trace_interpretation,
        "derivative_interpretation": derivative_interpretation,
        "baseline_default_value": baseline_default,
        "reporting_default_value": reporting_default,
        "derivative_pairs": derivative_pairs,
    }


def _deserialize_default_value(default_value):
    # verify type
    type_ = default_value.get('type', None)
    if type_ is None:
        return {
            'error': 'Serialization "type" must be provided for'
            ' baseline_default_value or reporting_default_value.'
        }

    # switch on type
    if type_ == 'SIMPLE_DEFAULT':
        return _deserialize_simple_default(default_value)
    else:
        return {
            'error': 'Serialization type "{}" not recognized.'.format(type_)
        }


def _deserialize_simple_default(default_value):
    try:
        return Derivative(
            None,
            default_value["value"],
            default_value["lower"],
            default_value["upper"],
            default_value["n"],
            None
        )
    except:
        return {
            "error": "Missing key in default_value serialization."
        }


def _deserialize_derivatives(derivatives):

    # verify type
    type_ = derivatives.get('type', None)
    if type_ is None:
        return {
            'error': 'Serialization "type" must be provided for'
            ' derivatives.'
        }

    # switch on type
    if type_ == 'DERIVATIVE_PAIRS':
        return _deserialize_derivative_pairs(derivatives)
    else:
        return {
            'error': 'Serialization type "{}" not recognized.'.format(type_)
        }


def _deserialize_derivative_pairs(derivatives):

    # check for "derivative_interpretation" key
    derivative_pairs, error = _get_key_or_error(
        derivatives, "derivative_pairs", "DERIVATIVE_PAIRS")
    if derivative_pairs is None:
        return error

    try:
        return [
            DerivativePair(
                pair["label"],
                pair["derivative_interpretation"],
                pair["trace_interpretation"],
                pair["unit"],
                Derivative(
                    None,
                    pair["baseline_value"],
                    pair["baseline_lower"],
                    pair["baseline_upper"],
                    pair["baseline_n"],
                    None
                ),
                Derivative(
                    None,
                    pair["reporting_value"],
                    pair["reporting_lower"],
                    pair["reporting_upper"],
                    pair["reporting_n"],
                    None
                ),
            ) for pair in derivative_pairs
        ]
    except KeyError:
        return {
            "error": "Missing key in derivative_pair serialization."
        }
