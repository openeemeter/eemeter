class ModelingPeriod(object):
    ''' Represents a period of time over which to select data from a Trace
    for contiguous modeling. Carries an "interpretation", for which there are
    two options, `"BASELINE"` and `"REPORTING"`. The period is defined by a
    single optional start date and a single optional end date. If the start
    date is not given, the start date is considered to be negative infinity;
    if the end date is not given, the end date is considered to be positive
    infinity.

    A ModelingPeriod is a time period, defined by start and end dates, over
    which the process behind a trace can be expected, for modeling purposes,
    to have roughly the same energy response to end use demand. Note that this
    criterion might not be particularly well specified without reference to a
    particular intervention and set of modeling conditions.

    Parameters
    ----------
    interpretation : str, {"BASELINE", "REPORTING"}
        The way this ModelingPeriod should be interpreted.

          - `"BASELINE"` means that this modeling period represents the time
            *before* an intervention or set of interventions.
          - `"REPORTING"` means that this modeling period represents the time
            *after* an intervention or set of interventions.

    start_date : datetime.datetime or None
        The date marking the earliest date of the ModelingPeriod. `None`
        indicates a start_date of negative infinity. If interpretation is
        "REPORTING", start_date cannot be `None`.
    end_date : datetime.datetime or None
        The date marking the latest date of the ModelingPeriod. `None`
        indicates an end_date of positive infinity. If interpretation is
        "BASELINE", end_date cannot be `None`.

    '''

    VALID_INTERPRETATIONS = [
        "BASELINE",
        "REPORTING",
    ]

    def __init__(self, interpretation, start_date=None, end_date=None):
        self.interpretation = self._validated_interpretation(interpretation)
        self.start_date = start_date
        self.end_date = end_date

        if start_date is None:
            if self.interpretation == "REPORTING":
                message = (
                    'For interpretation="REPORTING", start_date required.'
                    .format(interpretation, self.VALID_INTERPRETATIONS)
                )
                raise ValueError(message)
        else:
            self._validate_tz_aware(start_date)

        if end_date is None:
            if self.interpretation == "BASELINE":
                message = (
                    'For interpretation="BASELINE", end_date required.'
                    .format(interpretation, self.VALID_INTERPRETATIONS)
                )
                raise ValueError(message)
        else:
            self._validate_tz_aware(end_date)

        if start_date is not None and end_date is not None:
            self._validate_date_order(start_date, end_date)

    def _validated_interpretation(self, interpretation):
        if interpretation not in self.VALID_INTERPRETATIONS:
            message = (
                'Interpretation "{}" not recognized, use one of these: {}'
                .format(interpretation, self.VALID_INTERPRETATIONS)
            )
            raise ValueError(message)
        else:
            return interpretation

    def _validate_date_order(self, start_date, end_date):
        if start_date > end_date:
            message = (
                'Invalid dates: start_date({}) > end_date({}).'
                .format(start_date, end_date)
            )
            raise ValueError(message)

    def _validate_tz_aware(self, dt):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            message = (
                "Date must be timezone aware: {}".format(dt)
            )
            raise ValueError(message)

    def __repr__(self):
        return (
            'ModelingPeriod("{}", start_date={}, end_date={})'
            .format(self.interpretation, self.start_date, self.end_date)
        )


class ModelingPeriodSet(object):
    ''' Represents a set of labeled modeling periods of interest, grouped into
    meaningful comparison sets. Labels can be arbitrary.

    Basic usage:

    .. code-block:: python

        >>> modeling_periods = {
        ...     "modeling_period_1": ModelingPeriod(
        ...         "BASELINE",
        ...         end_date=datetime(2000, 1, 1, tzinfo=pytz.UTC),
        ...     ),
        ...     "modeling_period_2": ModelingPeriod(
        ...         "REPORTING",
        ...         start_date=datetime(2000, 2, 1, tzinfo=pytz.UTC),
        ...     ),
        ...     "modeling_period_3": ModelingPeriod(
        ...         "REPORTING",
        ...         start_date=datetime(2000, 2, 1, tzinfo=pytz.UTC),
        ...     ),
        ... }
        ...
        >>> grouping = [
        ...     ("modeling_period_1", "modeling_period_2"),
        ...     ("modeling_period_1", "modeling_period_3"),
        ... ]
        ...
        >>> mps = ModelingPeriodSet(modeling_periods, grouping)

    Parameters
    ---------
    '''

    def __init__(self, modeling_periods, groupings):
        self.modeling_periods = modeling_periods
        self.groupings = groupings
        self._validate()

    def __repr__(self):
        return (
            'ModelingPeriodSet(modeling_periods={}, groupings={})'
            .format(self.modeling_periods, self.groupings)
        )

    def _validate(self):
        for grouping in self.groupings:
            baseline_period, reporting_period = grouping
            self._validate_modeling_period(baseline_period, "BASELINE")
            self._validate_modeling_period(reporting_period, "REPORTING")

    def _validate_modeling_period(self, label, interpretation):
        # check that key exists
        if label not in self.modeling_periods:
            message = (
                "Key {} in `groupings` not found in"
                " `modeling_periods` dict. Available"
                " keys are {}."
                .format(label, list(self.modeling_periods.keys()))
            )
            raise ValueError(message)

        # get period and check that interpretation matches expected format.
        modeling_period = self.modeling_periods[label]
        if interpretation != modeling_period.interpretation:
            message = (
                'The interpretation given for ModelingPeriod "{}"'
                ' is not valid. It should be "{}".'
                .format(modeling_period.interpretation, interpretation)
            )
            raise ValueError(message)

    def iter_modeling_period_groups(self):
        for labels in self.groupings:
            baseline_label, reporting_label = labels
            modeling_periods = (
                self.modeling_periods[baseline_label],
                self.modeling_periods[reporting_label]
            )
            yield labels, modeling_periods

    def iter_modeling_periods(self):
        for label, modeling_period in sorted(self.modeling_periods.items(),
                                             key=lambda x: x[0]):
            yield label, modeling_period
