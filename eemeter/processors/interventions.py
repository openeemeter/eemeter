from eemeter.containers import ModelingPeriod
from eemeter.containers import ModelingPeriodSet


class EEModelingPeriodProcessor(object):

    def get_modeling_period_set(self, interventions):

        # don't attempt modeling where there are no interventions
        if len(interventions) == 0:
            return []

        baseline_period_end = self._get_earliest_intervention_start_date(
                interventions)
        reporting_period_start = self._get_latest_intervention_end_date(
                interventions)

        if reporting_period_start is None:
            # fall back to baseline_period_end - interventions are still
            # ongoing.
            reporting_period_start = baseline_period_end

        modeling_periods = {
            "baseline": ModelingPeriod("BASELINE",
                                       end_date=baseline_period_end),
            "reporting": ModelingPeriod("REPORTING",
                                        start_date=reporting_period_start),
        }

        groupings = [("baseline", "reporting")]

        modeling_period_set = ModelingPeriodSet(modeling_periods, groupings)

        validation_errors = []

        return modeling_period_set, validation_errors

    def _get_earliest_intervention_start_date(self, interventions):
        return min([i.start_date for i in interventions])

    def _get_latest_intervention_end_date(self, interventions):

        non_null_end_dates = [
            i.end_date for i in interventions if i.end_date is not None
        ]

        if len(non_null_end_dates) > 0:
            return max([d for d in non_null_end_dates])
        else:
            return None
