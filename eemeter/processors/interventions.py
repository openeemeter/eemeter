import logging

from eemeter.structures import (
    ModelingPeriod,
    ModelingPeriodSet,
)

logger = logging.getLogger(__name__)


def get_modeling_period_set(interventions):
    ''' Creates an applicable modeling period set given a list of
    interventions.

    Parameters
    ----------
    interventions : list of eemeter.structures.Intervention
        Interventions for which to build ModelingPeriodSet.
    '''

    # don't attempt modeling where there are no interventions
    if len(interventions) == 0:
        logger.info("No interventions, so no modeling period set.")
        return None

    baseline_period_end = _get_earliest_intervention_start_date(
        interventions)
    reporting_period_start = _get_latest_intervention_end_date(
        interventions)

    if reporting_period_start is None:
        # fall back to baseline_period_end - interventions are still
        # ongoing.
        reporting_period_start = baseline_period_end

    modeling_periods = {
        "baseline": ModelingPeriod(
            "BASELINE",
            end_date=baseline_period_end
        ),
        "reporting": ModelingPeriod(
            "REPORTING",
            start_date=reporting_period_start
        ),
    }

    groupings = [("baseline", "reporting")]

    modeling_period_set = ModelingPeriodSet(modeling_periods, groupings)

    logger.info("Created one modeling period group.")

    return modeling_period_set


def _get_earliest_intervention_start_date(interventions):
    return min([i.start_date for i in interventions])


def _get_latest_intervention_end_date(interventions):

    non_null_end_dates = [
        i.end_date for i in interventions if i.end_date is not None
    ]

    if len(non_null_end_dates) > 0:
        return max([d for d in non_null_end_dates])
    else:
        return None
