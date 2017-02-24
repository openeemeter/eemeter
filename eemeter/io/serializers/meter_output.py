from collections import OrderedDict


def serialize_derivatives(derivatives):
    return [serialize_derivative(d) for d in derivatives]


def serialize_derivative(d):
    return OrderedDict([
        ("modeling_period_group", d.modeling_period_group),
        ("series", d.series),
        ("description", d.description),
        ("orderable", d.orderable),
        ("value", d.value),
        ("variance", d.variance)
    ])


def serialize_split_modeled_energy_trace(modeled_trace):
    serialized = OrderedDict([
        ("type", "SPLIT_MODELED_ENERGY_TRACE"),
        ("fits", _serialize_fit_outputs(modeled_trace.fit_outputs)),
        ("modeling_period_set", serialize_modeling_period_set(
            modeled_trace.modeling_period_set)),
    ])

    return serialized


def _serialize_fit_outputs(fit_outputs):
    return OrderedDict([
        (mp_label, OrderedDict([
            ("status", output.get("status", None)),
            ("traceback", output.get("traceback", None)),
            ("start_date",
                None if output.get("start_date", None) is None
                else output.get("start_date").isoformat()),
            ("end_date",
                None if output.get("end_date", None) is None
                else output.get("end_date").isoformat()),
            ("n_rows", output.get("n_rows", None)),
            ("model_fit", _serialize_model_fit(output.get("model_fit", None))),
            ("input_data", output.get("input_data_serialization", None)),
        ]))
        for mp_label, output in sorted(fit_outputs.items(), key=lambda x: x[0])
    ])


def _serialize_model_fit(model_fit):
    if model_fit is None:
        return None
    else:
        model_params = {}
        for k, v in model_fit.get("model_params", {}).items():
            if isinstance(v, type({})):
                model_params[k] = OrderedDict(v)
            elif k == 'X_design_info':
                continue
            else:
                model_params[k] = v
        model_params = OrderedDict(model_params)
        return OrderedDict([
            ("r2", model_fit.get("r2", None)),
            ("cvrmse", model_fit.get("cvrmse", None)),
            ("rmse", model_fit.get("rmse", None)),
            ("lower", model_fit.get("lower", None)),
            ("upper", model_fit.get("upper", None)),
            ("n", model_fit.get("n", None)),
            ("model_params", model_params),
        ])


def serialize_modeling_period_set(modeling_period_set):
    return OrderedDict([
        ("modeling_period_groups", [
            OrderedDict([("baseline", g[0]), ("reporting", g[1])])
            for g in modeling_period_set.groupings
        ]),
        ("modeling_periods", OrderedDict([
            (label, OrderedDict([
                ("interpretation", mp.interpretation),
                ("start_date",
                    None if mp.start_date is None
                    else mp.start_date.isoformat()),
                ("end_date",
                    None if mp.end_date is None else mp.end_date.isoformat()),
            ]))
            for label, mp in sorted(
                modeling_period_set.modeling_periods.items(),
                key=lambda x: x[0])
        ])),
    ])
