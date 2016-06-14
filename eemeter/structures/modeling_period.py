class ModelingPeriod(object):

    INTERPRETATIONS = [
        "BASELINE",
        "REPORTING",
    ]

    def __init__(self, interpretation, start_date=None, end_date=None):
        self.interpretation = self._validated_interpretation(interpretation)
        self.start_date = start_date
        self.end_date = end_date

    def _validated_interpretation(self, interpretation):
        if interpretation not in self.INTERPRETATIONS:
            message = (
                'Interpretation "{}" not recognized, use on of: {}'
                .format(interpretation, self.INTERPRETATIONS)
            )
            raise ValueError(message)
        else:
            return interpretation

    def __repr__(self):
        return (
            'ModelingPeriod("{}", {} <-> {})'
            .format(self.interpretation, self.start_date, self.end_date)
        )


class ModelingPeriodSet(object):

    def __init__(self, modeling_periods, groupings):
        self.modeling_periods = modeling_periods
        self.groupings = groupings
        self._validate()

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
                "  `modeling_periods` dict. Available"
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

    def get_modeling_period_groups(self):
        for labels in self.groupings:
            baseline_label, reporting_label = labels
            yield (
                (baseline_label, self.modeling_periods[baseline_label]),
                (reporting_label, self.modeling_periods[reporting_label]),
            )

    def get_modeling_periods(self):
        for label, modeling_period in self.modeling_periods.items():
            yield label, modeling_period
