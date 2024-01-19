#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2023 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
from dateutil.relativedelta import relativedelta

from .design_matrices import (
    create_caltrack_hourly_preliminary_design_matrix,
    create_caltrack_hourly_segmented_design_matrices,
    create_caltrack_daily_design_matrix,
    create_caltrack_billing_design_matrix,
)
from .hourly import fit_caltrack_hourly_model
from .usage_per_day import caltrack_sufficiency_criteria
from ..transform import get_baseline_data, get_reporting_data, clean_caltrack_billing_daily_data
from ..derivatives import metered_savings
from ..features import estimate_hour_of_week_occupancy, fit_temperature_bins, compute_temperature_features
from ..segmentation import segment_time_series

from eemeter.eemeter.models import DailyModel
import pandas as pd

__all__ = (
    "caltrack_daily",
    "caltrack_hourly",
)


def caltrack_hourly(
    meter_data,
    temperature_data,
    blackout_start_date,
    blackout_end_date,
    degc: bool = False,
):
    """An output function which takes meter data, external temperature data, blackout start and end dates, and
    returns a metered savings dataframe for the period between the blackout end date and today.

    Parameters
    ----------
    meter_data : :any:`pandas.DataFrame`
        Hourly series meter data, unit kWh.
    temperature_data : :any:``
        Hourly external temperature data. If DataFrame, not pd.Series (as required by CalTRACK) function will convert.
    blackout_start_date : :any: 'datetime.datetime'
        The date at which improvement works commenced.
    blackout_end_date : :any: 'datetime.datetime'
        The date by which improvement works completed and metering resumed.
    degc : :any 'bool'
        Relevant temperature units; defaults to False (i.e. Fahrenheit).

    Returns
    -------
    metered_savings_dataframe: :any:`pandas.DataFrame`
    DataFrame with metered savings, indexed with
    ``reporting_meter_data.index``. Will include the following columns:

     - ``counterfactual_usage`` (baseline model projected into reporting period)
     - ``reporting_observed`` (given by reporting_meter_data)
     - ``metered_savings``

     If `with_disaggregated` is set to True, the following columns will also
     be in the results DataFrame:

     - ``counterfactual_base_load``
     - ``counterfactual_heating_load``
     - ``counterfactual_cooling_load``
    """

    baseline_meter_data, baseline_warnings = get_baseline_data(
        meter_data,
        start=blackout_start_date - relativedelta(years=1),
        end=blackout_start_date,
        max_days=None,
    )

    # create a design matrix for occupancy and segmentation
    preliminary_design_matrix = create_caltrack_hourly_preliminary_design_matrix(
        baseline_meter_data, temperature_data, degc
    )

    # build 12 monthly models - each step from now on operates on each segment
    segmentation = segment_time_series(
        preliminary_design_matrix.index, "three_month_weighted"
    )

    # assign an occupancy status to each hour of the week (0-167)
    occupancy_lookup = estimate_hour_of_week_occupancy(
        preliminary_design_matrix, segmentation=segmentation
    )

    # assign temperatures to bins
    (
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    ) = fit_temperature_bins(
        preliminary_design_matrix,
        segmentation=segmentation,
        occupancy_lookup=occupancy_lookup,
    )

    # build a design matrix for each monthly segment
    segmented_design_matrices = create_caltrack_hourly_segmented_design_matrices(
        preliminary_design_matrix,
        segmentation,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )

    # build a CalTRACK hourly model
    baseline_model = fit_caltrack_hourly_model(
        segmented_design_matrices,
        occupancy_lookup,
        occupied_temperature_bins,
        unoccupied_temperature_bins,
    )
    # get a year of reporting period data
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=365
    )

    # compute metered savings for the year of the reporting period we've selected
    metered_savings_dataframe, error_bands = metered_savings(
        baseline_model,
        reporting_meter_data,
        temperature_data,
        with_disaggregated=True,
        degc=degc,
    )

    return metered_savings_dataframe

def caltrack_daily(
    meter_data,
    temperature_data,
    blackout_start_date,
    blackout_end_date,
):
    """An output function which takes meter data, external temperature data, blackout start and end dates, and
       returns a metered savings dataframe for the period between the blackout end date and today. Note that
       the daily model is indifferent to temperature units, as long as they are kept consistent during fit and prediction.

       Parameters
       ----------
       meter_data : :any:`pandas.DataFrame`
           Daily series meter data, unit kWh.
       temperature_data : :any:``
           Hourly external temperature data. If DataFrame, not pd.Series (as required by CalTRACK) function will convert.
       blackout_start_date : :any: 'datetime.datetime'
           The date at which improvement works commenced.
       blackout_end_date : :any: 'datetime.datetime'
           The date by which improvement works completed and metering resumed.

       Returns
       -------
       metered_savings_dataframe: :any:`pandas.DataFrame`
       DataFrame with metered savings, indexed with
       ``reporting_meter_data.index``. Will include the following columns:

        - ``counterfactual_usage`` (baseline model projected into reporting period)
        - ``reporting_observed`` (given by reporting_meter_data)
        - ``metered_savings``

        If `with_disaggregated` is set to True, the following columns will also
        be in the results DataFrame:

        - ``counterfactual_base_load``
        - ``counterfactual_heating_load``
        - ``counterfactual_cooling_load``
       """
    # get 365 days of baseline data, ending at blackout_start_date
    baseline_meter_data, warnings = get_baseline_data(
        meter_data,
        end=blackout_start_date,
    ) 
    baseline_meter_dataframe = create_caltrack_daily_design_matrix(baseline_meter_data, temperature_data)
    
    # check data sufficiency
    sc = caltrack_sufficiency_criteria(
        baseline_meter_dataframe,
        requested_start=blackout_start_date - relativedelta(days=365),
        requested_end=blackout_start_date,
        num_days=365,
    )
    if sc.status != 'PASS':
        # raise exception, etc
        print(sc.data)
    baseline_model = DailyModel().fit(baseline_meter_dataframe)

    # get all reporting data after blackout_end_date
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, max_days=None
    )

    # compute metered savings for the year of the reporting period we've selected
    metered_savings_dataframe, error_bands = metered_savings(
        baseline_model,
        reporting_meter_data,
        temperature_data,
        with_disaggregated=True,
    )

    return metered_savings_dataframe

def caltrack_billing(
    meter_data,
    temperature_data,
    blackout_start_date,
    blackout_end_date,
):
    """An output function which takes meter data, external temperature data, blackout start and end dates, and
       returns a metered savings dataframe for the period between the blackout end date and today. Note that
       the daily model is indifferent to temperature units, as long as they are kept consistent during fit and prediction.

       Parameters
       ----------
       meter_data : :any:`pandas.DataFrame`
           Billing meter data, unit kWh.
       temperature_data : :any:``
           Hourly external temperature data. If DataFrame, not pd.Series (as required by CalTRACK) function will convert.
       blackout_start_date : :any: 'datetime.datetime'
           The date at which improvement works commenced.
       blackout_end_date : :any: 'datetime.datetime'
           The date by which improvement works completed and metering resumed.

       Returns
       -------
       metered_savings_dataframe: :any:`pandas.DataFrame`
       DataFrame with metered savings, indexed with
       ``reporting_meter_data.index``. Will include the following columns:

        - ``counterfactual_usage`` (baseline model projected into reporting period)
        - ``reporting_observed`` (given by reporting_meter_data)
        - ``metered_savings``

        If `with_disaggregated` is set to True, the following columns will also
        be in the results DataFrame:

        - ``counterfactual_base_load``
        - ``counterfactual_heating_load``
        - ``counterfactual_cooling_load``
       """
    # get 365 days of baseline data, ending at blackout_start_date
    baseline_meter_data, warnings = get_baseline_data(
        meter_data,
        end=blackout_start_date,
        allow_billing_period_overshoot=True,
    )

    # create design matrix in order to handle usage-per-day calculations and temperature avgs
    baseline_meter_dataframe = create_caltrack_billing_design_matrix(baseline_meter_data, temperature_data)

    # fit daily model on baseline data
    baseline_model = DailyModel().fit(baseline_meter_dataframe)

    # get one year of reporting data after blackout_end_date
    reporting_meter_data, warnings = get_reporting_data(
        meter_data, start=blackout_end_date, allow_billing_period_overshoot=True
    )

    # compute metered savings for the year of the reporting period we've selected
    metered_savings_dataframe, error_bands = metered_savings(
        baseline_model,
        reporting_meter_data,
        temperature_data,
        with_disaggregated=True,
        billing_data=True,
    )
    return metered_savings_dataframe