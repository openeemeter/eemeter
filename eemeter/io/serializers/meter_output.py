from collections import OrderedDict

def serialize_derivative_pairs(derivative_pairs):
    serialized = []
    for interpretation, baseline, reporting in derivative_pairs:

        if baseline is not None:
            baseline_serialized = OrderedDict([
                ("label", baseline.label),
                ("value", baseline.value),
                ("lower", baseline.lower),
                ("upper", baseline.upper),
                ("n", baseline.n),
            ])
        else:
            baseline_serialized = None

        if reporting is not None:
            reporting_serialized = OrderedDict([
                ("label", reporting.label),
                ("value", reporting.value),
                ("lower", reporting.lower),
                ("upper", reporting.upper),
                ("n", reporting.n),
            ])
        else:
            reporting_serialized = None

        serialized.append(OrderedDict([
            ("interpretation", interpretation),
            ("baseline", baseline_serialized),
            ("reporting", reporting_serialized),
        ]))
    return serialized
