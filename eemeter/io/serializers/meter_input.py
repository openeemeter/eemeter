from .trace import (
    ArbitrarySerializer,
    ArbitraryStartSerializer,
    ArbitraryEndSerializer,
)
from eemeter.structures import (
    EnergyTrace,
    ModelingPeriod,
    ModelingPeriodSet
)

import dateutil.parser


def deserialize_meter_input(meter_input):

    # verify type
    type_ = meter_input.get('type', None)
    if type_ is None:
        return {
            'error': 'Serialization "type" must be provided for meter_input.'
        }

    # switch on type
    if type_ == 'SINGLE_TRACE_SIMPLE_PROJECT':
        return _deserialize_single_trace_simple_project(meter_input)
    else:
        return {
            'error': 'Serialization type "{}" not recognized.'.format(type_)
        }


def _deserialize_single_trace_simple_project(meter_input):

    # check for "trace" key
    single_trace = meter_input.get('trace', None)
    if single_trace is None:
        return {
            'error': (
                'For serialization type "SINGLE_TRACE_SIMPLE_PROJECT",'
                ' key "trace" must be provided.'
            )
        }

    # check for "project" key
    simple_project = meter_input.get('project', None)
    if simple_project is None:
        return {
            'error': (
                'For serialization type "SINGLE_TRACE_SIMPLE_PROJECT",'
                ' key "project" must be provided.'
            )
        }

    trace = _deserialize_single_trace(single_trace)
    if "error" in trace:
        return trace

    project = _deserialize_simple_project(simple_project)
    if "error" in project:
        return project

    return {
        "trace": trace["trace"],
        "project": project["project"],
    }


def _deserialize_single_trace(trace):

    # verify type
    type_ = trace.get('type', None)
    if type_ is None:
        return {
            'error': 'Serialization "type" not given for trace.'
        }

    # check for "interpretation" key
    interpretation = trace.get('interpretation', None)
    if interpretation is None:
        return {
            'error': (
                'Trace serializations must provide key "interpretation".'
            )
        }

    # check for "unit" key
    unit = trace.get('unit', None)
    if unit is None:
        return {
            'error': (
                'Trace serializations must provide key "unit".'
            )
        }

    # check for "records" key
    records = trace.get('records', None)
    if records is None:
        return {
            'error': (
                'Trace serializations must provide key "records".'
            )
        }

    # check for optional "trace_id" key
    trace_id = trace.get('trace_id', None)

    # check for optional "interval" key
    interval = trace.get('interval', None)

    # switch on type
    if type_ == 'ARBITRARY':
        return {
            "trace": EnergyTrace(
                interpretation=interpretation,
                unit=unit,
                records=records,
                serializer=ArbitrarySerializer(parse_dates=True),
                trace_id=trace_id,
                interval=interval,
            )
        }
    elif type_ == 'ARBITRARY_START':
        return {
            "trace": EnergyTrace(
                interpretation=interpretation,
                unit=unit,
                records=records,
                serializer=ArbitraryStartSerializer(parse_dates=True),
                trace_id=trace_id,
                interval=interval,
            )
        }
    elif type_ == 'ARBITRARY_END':
        return {
            "trace": EnergyTrace(
                interpretation=interpretation,
                unit=unit,
                records=records,
                serializer=ArbitraryEndSerializer(parse_dates=True),
                trace_id=trace_id,
                interval=interval,
            )
        }
    else:
        return {
            'error': (
                'Serialization type "{}" not recognized for trace.'
                .format(type_)
            )
        }


def _deserialize_simple_project(project):
    # verify type
    type_ = project.get('type', None)
    if type_ is None:
        return {
            'error': 'Serialization "type" not given for project.'
        }

    if type_ == "PROJECT_WITH_SINGLE_MODELING_PERIOD_GROUP":
        return _deserialize_project_with_single_model_period_group(project)
    else:
        return {
            'error': (
                'Serialization type "{}" not recognized for project.'
                .format(type_)
            )
        }


def _deserialize_project_with_single_model_period_group(project):

    # check for "zipcode" key
    zipcode = project.get('zipcode', None)
    if zipcode is None:
        return {
            'error': (
                'Project serializations must provide key "zipcode".'
            )
        }

    # check for "modeling_period_group" key
    modeling_period_group = project.get('modeling_period_group', None)
    if modeling_period_group is None:
        return {
            'error': (
                'Project serializations must provide key'
                ' "modeling_period_group".'
            )
        }

    modeling_period_set = _deserialize_single_modeling_period_group(
        modeling_period_group
    )
    if "error" in modeling_period_set:
        return modeling_period_set

    # check for optional "project_id" key
    project_id = project.get('project_id', None)

    return {
        "project": {
            "zipcode": zipcode,
            "modeling_period_set": modeling_period_set["modeling_period_set"],
            "project_id": project_id,
        }
    }


def _deserialize_single_modeling_period_group(modeling_period_group):

    # check for "baseline_period" key
    baseline_period = modeling_period_group.get('baseline_period', None)
    if baseline_period is None:
        return {
            'error': (
                'Project serializations must provide key'
                ' "baseline_period".'
            )
        }
    else:
        start_date = baseline_period.get("start", None)
        if start_date is not None:
            start_date = dateutil.parser.parse(start_date)

        end_date = baseline_period.get("end", None)
        if end_date is not None:
            end_date = dateutil.parser.parse(end_date)

        baseline = ModelingPeriod(
            "BASELINE",
            start_date,
            end_date
        )

    # check for "reporting_period" key
    reporting_period = modeling_period_group.get('reporting_period', None)
    if reporting_period is None:
        return {
            'error': (
                'Project serializations must provide key'
                ' "reporting_period".'
            )
        }
    else:
        start_date = reporting_period.get("start", None)
        if start_date is not None:
            start_date = dateutil.parser.parse(start_date)

        end_date = reporting_period.get("end", None)
        if end_date is not None:
            end_date = dateutil.parser.parse(end_date)

        reporting = ModelingPeriod(
            "REPORTING",
            start_date,
            end_date
        )

    modeling_periods = {
        "baseline": baseline,
        "reporting": reporting,
    }

    grouping = [
        ("baseline", "reporting"),
    ]

    mps = ModelingPeriodSet(modeling_periods, grouping)
    return {
        "modeling_period_set": mps
    }
