from collections import Counter

import numpy as np
import pandas as pd
import pytz
import statsmodels.formula.api as smf
import traceback

from .api import CandidateModel, DataSufficiency, EEMeterWarning, ModelResults
from .exceptions import MissingModelParameterError, UnrecognizedModelTypeError
from .transform import (
    day_counts,
    compute_temperature_features,
    overwrite_partial_rows_with_nan,
)
from .metrics import ModelMetrics


__all__ = (
    "caltrack_method",
    "caltrack_sufficiency_criteria",
    "caltrack_metered_savings",
    "caltrack_modeled_savings",
    "caltrack_predict",
    "plot_caltrack_candidate",
    "get_too_few_non_zero_degree_day_warning",
    "get_total_degree_day_too_low_warning",
    "get_parameter_negative_warning",
    "get_parameter_p_value_too_high_warning",
    "get_single_cdd_only_candidate_model",
    "get_single_hdd_only_candidate_model",
    "get_single_cdd_hdd_candidate_model",
    "get_intercept_only_candidate_models",
    "get_cdd_only_candidate_models",
    "get_hdd_only_candidate_models",
    "get_cdd_hdd_candidate_models",
    "select_best_candidate",
)


def _candidate_model_factory(
    model_type,
    formula,
    status,
    warnings=None,
    model_params=None,
    model=None,
    result=None,
    r_squared_adj=None,
    use_predict_func=True,
):
    if use_predict_func:
        predict_func = caltrack_predict
    else:
        predict_func = None

    return CandidateModel(
        model_type=model_type,
        formula=formula,
        status=status,
        warnings=warnings,
        predict_func=predict_func,
        plot_func=plot_caltrack_candidate,
        model_params=model_params,
        model=model,
        result=result,
        r_squared_adj=r_squared_adj,
    )


def _get_parameter_or_raise(model_type, model_params, param):
    try:
        return model_params[param]
    except KeyError:
        raise MissingModelParameterError(
            '"{}" parameter required for model_type: {}'.format(param, model_type)
        )


def _caltrack_predict_design_matrix(
    model_type,
    model_params,
    data,
    disaggregated=False,
    input_averages=False,
    output_averages=False,
):
    """ An internal CalTRACK predict method for use with a design matrix of the form
    used in model fitting.

    Given a set model type, parameters, and daily temperatures, return model
    predictions.

    Parameters
    ----------
    model_type : :any:`str`
        Model type (e.g., ``'cdd_hdd'``).
    model_params : :any:`dict`
        Parameters as stored in :any:`eemeter.CandidateModel.model_params`.
    data : :any:`pandas.DataFrame`
        Data over which to predict. Assumed to be like the format of the data used
        for fitting, although it need only have the columns. If not giving data
        with a `pandas.DatetimeIndex` it must have the column `n_days`,
        representing the number of days per prediction period (otherwise
        inferred from DatetimeIndex).
    disaggregated : :any:`bool`, optional
        If True, return results as a :any:`pandas.DataFrame` with columns
        ``'base_load'``, ``'heating_load'``, and ``'cooling_load'``
    input_averages : :any:`bool`, optional
        If HDD and CDD columns expressed as period totals, select False. If HDD
        and CDD columns expressed as period averages, select True. If prediction
        period is daily, results should be the same either way. Matters for billing.
    output_averages : :any:`bool`, optional
        If True, prediction returned as averages (not totals). If False, returned
        as totals.

    Returns
    -------
    prediction : :any:`pandas.Series` or :any:`pandas.DataFrame`
        Returns results as series unless ``disaggregated=True``.
    """

    zeros = pd.Series(0, index=data.index)
    ones = zeros + 1

    if isinstance(data.index, pd.DatetimeIndex):
        days_per_period = day_counts(zeros)

    # TODO(philngo): handle different degree day methods and hourly temperatures
    if model_type in ["intercept_only", "hdd_only", "cdd_only", "cdd_hdd"]:
        intercept = _get_parameter_or_raise(model_type, model_params, "intercept")
        if output_averages == False:
            base_load = intercept * days_per_period
        else:
            base_load = intercept * ones
        # The last row of data was nan -- Restore the NaN
        base_load[-1] = np.nan
    elif model_type is None:
        raise ValueError("Model not valid for prediction: model_type=None")
    else:
        raise UnrecognizedModelTypeError(
            "invalid caltrack model type: {}".format(model_type)
        )

    if model_type in ["hdd_only", "cdd_hdd"]:
        beta_hdd = _get_parameter_or_raise(model_type, model_params, "beta_hdd")
        heating_balance_point = _get_parameter_or_raise(
            model_type, model_params, "heating_balance_point"
        )
        hdd_column_name = "hdd_%s" % heating_balance_point
        hdd = data[hdd_column_name]
        if input_averages == True and output_averages == False:
            heating_load = hdd * beta_hdd * days_per_period
        elif input_averages == True and output_averages == True:
            heating_load = hdd * beta_hdd
        elif input_averages == False and output_averages == False:
            heating_load = hdd * beta_hdd
        else:
            heating_load = hdd * beta_hdd / days_per_period
    else:
        heating_load = zeros

    if model_type in ["cdd_only", "cdd_hdd"]:
        beta_cdd = _get_parameter_or_raise(model_type, model_params, "beta_cdd")
        cooling_balance_point = _get_parameter_or_raise(
            model_type, model_params, "cooling_balance_point"
        )
        cdd_column_name = "cdd_%s" % cooling_balance_point
        cdd = data[cdd_column_name]
        if input_averages == True and output_averages == False:
            cooling_load = cdd * beta_cdd * days_per_period
        elif input_averages == True and output_averages == True:
            cooling_load = cdd * beta_cdd
        elif input_averages == False and output_averages == False:
            cooling_load = cdd * beta_cdd
        else:
            cooling_load = cdd * beta_cdd / days_per_period
    else:
        cooling_load = zeros

    if disaggregated:
        return pd.DataFrame(
            {
                "base_load": base_load,
                "heating_load": heating_load,
                "cooling_load": cooling_load,
            }
        )
    else:
        return base_load + heating_load + cooling_load


def caltrack_predict(
    model_type,
    model_params,
    temperature_data,
    prediction_index,
    degree_day_method,
    with_disaggregated=False,
    with_design_matrix=False,
):
    """ CalTRACK predict method.

    Given a model type, parameters, hourly temperatures, a
    :any:`pandas.DatetimeIndex` index over which to predict meter usage,
    return model predictions as totals for the period (so billing period totals,
    daily totals, etc.). Optionally include the computed design matrix or
    disaggregated usage in the output dataframe.

    Parameters
    ----------
    model_type : :any:`str`
        Model type (e.g., ``'cdd_hdd'``).
    model_params : :any:`dict`
        Parameters as stored in :any:`eemeter.CandidateModel.model_params`.
    temperature_data : :any:`pandas.DataFrame`
        Hourly temperature data to use for prediction. Time period should match
        the ``prediction_index`` argument.
    prediction_index : :any:`pandas.DatetimeIndex`
        Time period over which to predict.
    with_disaggregated : :any:`bool`, optional
        If True, return results as a :any:`pandas.DataFrame` with columns
        ``'base_load'``, ``'heating_load'``, and ``'cooling_load'``.
    with_design_matrix : :any:`bool`, optional
        If True, return results as a :any:`pandas.DataFrame` with columns
        ``'n_days'``, ``'n_days_dropped'``, ``n_days_kept``, and
        ``temperature_mean``.

    Returns
    -------
    prediction : :any:`pandas.DataFrame`
        Columns are as follows:

        - ``predicted_usage``: Predicted usage values computed to match
          ``prediction_index``.
        - ``base_load``: modeled base load (only for ``with_disaggregated=True``).
        - ``cooling_load``: modeled cooling load (only for ``with_disaggregated=True``).
        - ``heating_load``: modeled heating load (only for ``with_disaggregated=True``).
        - ``n_days``: number of days in period (only for ``with_design_matrix=True``).
        - ``n_days_dropped``: number of days dropped because of insufficient
          data (only for ``with_design_matrix=True``).
        - ``n_days_kept``: number of days kept because of sufficient data
          (only for ``with_design_matrix=True``).
        - ``temperature_mean``: mean temperature during given period.
          (only for ``with_design_matrix=True``).
    """
    if model_params is None:
        raise MissingModelParameterError("model_params is None.")

    cooling_balance_points = []
    heating_balance_points = []

    if "cooling_balance_point" in model_params:
        cooling_balance_points.append(model_params["cooling_balance_point"])
    if "heating_balance_point" in model_params:
        heating_balance_points.append(model_params["heating_balance_point"])

    design_matrix = compute_temperature_features(
        temperature_data,
        prediction_index,
        heating_balance_points=heating_balance_points,
        cooling_balance_points=cooling_balance_points,
        degree_day_method=degree_day_method,
        use_mean_daily_values=False,
    )

    if design_matrix.empty:
        if with_disaggregated:
            empty_columns = {
                "predicted_usage": [],
                "base_load": [],
                "heating_load": [],
                "cooling_load": [],
            }
        else:
            empty_columns = {"predicted_usage": []}
        return pd.DataFrame(empty_columns, index=prediction_index)

    if degree_day_method == "daily":
        design_matrix["n_days"] = (
            design_matrix.n_days_kept + design_matrix.n_days_dropped
        )
    else:
        design_matrix["n_days"] = (
            design_matrix.n_hours_kept + design_matrix.n_hours_dropped
        ) / 24

    results = _caltrack_predict_design_matrix(
        model_type,
        model_params,
        design_matrix,
        input_averages=False,
        output_averages=False,
    ).to_frame("predicted_usage")

    if with_disaggregated:
        disaggregated = _caltrack_predict_design_matrix(
            model_type,
            model_params,
            design_matrix,
            disaggregated=True,
            input_averages=False,
            output_averages=False,
        )
        results = results.join(disaggregated)

    if with_design_matrix:
        results = results.join(design_matrix)

    return results


def get_too_few_non_zero_degree_day_warning(
    model_type, balance_point, degree_day_type, degree_days, minimum_non_zero
):
    """ Return an empty list or a single warning wrapped in a list regarding
    non-zero degree days for a set of degree days.

    Parameters
    ----------
    model_type : :any:`str`
        Model type (e.g., ``'cdd_hdd'``).
    balance_point : :any:`float`
        The balance point in question.
    degree_day_type : :any:`str`
        The type of degree days (``'cdd'`` or ``'hdd'``).
    degree_days : :any:`pandas.Series`
        A series of degree day values.
    minimum_non_zero : :any:`int`
        Minimum allowable number of non-zero degree day values.

    Returns
    -------
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        Empty list or list of single warning.
    """
    warnings = []
    n_non_zero = int((degree_days > 0).sum())
    if n_non_zero < minimum_non_zero:
        warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_daily.{model_type}.too_few_non_zero_{degree_day_type}".format(
                        model_type=model_type, degree_day_type=degree_day_type
                    )
                ),
                description=(
                    "Number of non-zero daily {degree_day_type} values below accepted minimum."
                    " Candidate fit not attempted.".format(
                        degree_day_type=degree_day_type.upper()
                    )
                ),
                data={
                    "n_non_zero_{degree_day_type}".format(
                        degree_day_type=degree_day_type
                    ): n_non_zero,
                    "minimum_non_zero_{degree_day_type}".format(
                        degree_day_type=degree_day_type
                    ): minimum_non_zero,
                    "{degree_day_type}_balance_point".format(
                        degree_day_type=degree_day_type
                    ): balance_point,
                },
            )
        )
    return warnings


def get_total_degree_day_too_low_warning(
    model_type, balance_point, degree_day_type, degree_days, minimum_total
):
    """ Return an empty list or a single warning wrapped in a list regarding
    the total summed degree day values.

    Parameters
    ----------
    model_type : :any:`str`
        Model type (e.g., ``'cdd_hdd'``).
    balance_point : :any:`float`
        The balance point in question.
    degree_day_type : :any:`str`
        The type of degree days (``'cdd'`` or ``'hdd'``).
    degree_days : :any:`pandas.Series`
        A series of degree day values.
    minimum_total : :any:`float`
        Minimum allowable total sum of degree day values.

    Returns
    -------
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        Empty list or list of single warning.
    """

    warnings = []
    total_degree_days = degree_days.sum()
    if total_degree_days < minimum_total:
        warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_daily.{model_type}.total_{degree_day_type}_too_low".format(
                        model_type=model_type, degree_day_type=degree_day_type
                    )
                ),
                description=(
                    "Total {degree_day_type} below accepted minimum."
                    " Candidate fit not attempted.".format(
                        degree_day_type=degree_day_type.upper()
                    )
                ),
                data={
                    "total_{degree_day_type}".format(
                        degree_day_type=degree_day_type
                    ): total_degree_days,
                    "total_{degree_day_type}_minimum".format(
                        degree_day_type=degree_day_type
                    ): minimum_total,
                    "{degree_day_type}_balance_point".format(
                        degree_day_type=degree_day_type
                    ): balance_point,
                },
            )
        )
    return warnings


def get_parameter_negative_warning(model_type, model_params, parameter):
    """ Return an empty list or a single warning wrapped in a list indicating
    whether model parameter is negative.

    Parameters
    ----------
    model_type : :any:`str`
        Model type (e.g., ``'cdd_hdd'``).
    model_params : :any:`dict`
        Parameters as stored in :any:`eemeter.CandidateModel.model_params`.
    parameter : :any:`str`
        The name of the parameter, e.g., ``'intercept'``.

    Returns
    -------
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        Empty list or list of single warning.
    """
    warnings = []
    if model_params.get(parameter, 0) < 0:
        warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_daily.{model_type}.{parameter}_negative".format(
                        model_type=model_type, parameter=parameter
                    )
                ),
                description=(
                    "Model fit {parameter} parameter is negative. Candidate model rejected.".format(
                        parameter=parameter
                    )
                ),
                data=model_params,
            )
        )
    return warnings


def get_parameter_p_value_too_high_warning(
    model_type, model_params, parameter, p_value, maximum_p_value
):
    """ Return an empty list or a single warning wrapped in a list indicating
    whether model parameter p-value is too high.

    Parameters
    ----------
    model_type : :any:`str`
        Model type (e.g., ``'cdd_hdd'``).
    model_params : :any:`dict`
        Parameters as stored in :any:`eemeter.CandidateModel.model_params`.
    parameter : :any:`str`
        The name of the parameter, e.g., ``'intercept'``.
    p_value : :any:`float`
        The p-value of the parameter.
    maximum_p_value : :any:`float`
        The maximum allowable p-value of the parameter.

    Returns
    -------
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        Empty list or list of single warning.
    """
    warnings = []
    if p_value > maximum_p_value:
        data = {
            "{}_p_value".format(parameter): p_value,
            "{}_maximum_p_value".format(parameter): maximum_p_value,
        }
        data.update(model_params)
        warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_daily.{model_type}.{parameter}_p_value_too_high".format(
                        model_type=model_type, parameter=parameter
                    )
                ),
                description=(
                    "Model fit {parameter} p-value is too high. Candidate model rejected.".format(
                        parameter=parameter
                    )
                ),
                data=data,
            )
        )
    return warnings


def get_fit_failed_candidate_model(model_type, formula):
    """ Return a Candidate model that indicates the fitting routine failed.

    Parameters
    ----------
    model_type : :any:`str`
        Model type (e.g., ``'cdd_hdd'``).
    formula : :any:`float`
        The candidate model formula.

    Returns
    -------
    candidate_model : :any:`eemeter.CandidateModel`
        Candidate model instance with status ``'ERROR'``, and warning with
        traceback.
    """
    warnings = [
        EEMeterWarning(
            qualified_name="eemeter.caltrack_daily.{}.model_results".format(model_type),
            description=(
                "Error encountered in statsmodels.formula.api.ols method. (Empty data?)"
            ),
            data={"traceback": traceback.format_exc()},
        )
    ]
    return _candidate_model_factory(model_type, formula, "ERROR", warnings)


def get_intercept_only_candidate_models(data, weights_col):
    """ Return a list of a single candidate intercept-only model.

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value``.
        DataFrames of this form can be made using the
        :any:`eemeter.merge_temperature_data` method.
    weights_col : :any:`str` or None
        The name of the column (if any) in ``data`` to use as weights.

    Returns
    -------
    candidate_models : :any:`list` of :any:`CandidateModel`
        List containing a single intercept-only candidate model.
    """
    model_type = "intercept_only"
    formula = "meter_value ~ 1"

    if weights_col is None:
        weights = 1
    else:
        weights = data[weights_col]

    try:
        model = smf.wls(formula=formula, data=data, weights=weights)
    except Exception as e:
        return [get_fit_failed_candidate_model(model_type, formula)]

    result = model.fit()

    # CalTrack 3.3.1.3
    model_params = {"intercept": result.params["Intercept"]}

    model_warnings = []

    # CalTrack 3.4.3.2
    for parameter in ["intercept"]:
        model_warnings.extend(
            get_parameter_negative_warning(model_type, model_params, parameter)
        )

    if len(model_warnings) > 0:
        status = "DISQUALIFIED"
    else:
        status = "QUALIFIED"

    return [
        _candidate_model_factory(
            model_type,
            formula,
            status,
            warnings=model_warnings,
            model_params=model_params,
            model=model,
            result=result,
            r_squared_adj=0,
        )
    ]


def get_single_cdd_only_candidate_model(
    data,
    minimum_non_zero_cdd,
    minimum_total_cdd,
    beta_cdd_maximum_p_value,
    weights_col,
    balance_point,
):
    """ Return a single candidate cdd-only model for a particular balance
    point.

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and
        ``cdd_<balance_point>``
        DataFrames of this form can be made using the
        :any:`eemeter.merge_temperature_data` method.
    minimum_non_zero_cdd : :any:`int`
        Minimum allowable number of non-zero cooling degree day values.
    minimum_total_cdd : :any:`float`
        Minimum allowable total sum of cooling degree day values.
    beta_cdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta cdd parameter.
    weights_col : :any:`str` or None
        The name of the column (if any) in ``data`` to use as weights.
    balance_point : :any:`float`
        The cooling balance point for this model.

    Returns
    -------
    candidate_model : :any:`CandidateModel`
        A single cdd-only candidate model, with any associated warnings.
    """
    model_type = "cdd_only"
    cdd_column = "cdd_%s" % balance_point
    formula = "meter_value ~ %s" % cdd_column

    degree_day_warnings = []
    degree_day_warnings.extend(
        get_total_degree_day_too_low_warning(
            model_type, balance_point, "cdd", data[cdd_column], minimum_total_cdd
        )
    )
    degree_day_warnings.extend(
        get_too_few_non_zero_degree_day_warning(
            model_type, balance_point, "cdd", data[cdd_column], minimum_non_zero_cdd
        )
    )

    if len(degree_day_warnings) > 0:
        return _candidate_model_factory(
            model_type,
            formula,
            "NOT ATTEMPTED",
            warnings=degree_day_warnings,
            use_predict_func=False,
        )

    if weights_col is None:
        weights = 1
    else:
        weights = data[weights_col]

    try:
        model = smf.wls(formula=formula, data=data, weights=weights)
    except Exception as e:
        return get_fit_failed_candidate_model(model_type, formula)

    result = model.fit()
    r_squared_adj = result.rsquared_adj
    beta_cdd_p_value = result.pvalues[cdd_column]

    # CalTrack 3.3.1.3
    model_params = {
        "intercept": result.params["Intercept"],
        "beta_cdd": result.params[cdd_column],
        "cooling_balance_point": balance_point,
    }

    model_warnings = []

    # CalTrack 3.4.3.2
    for parameter in ["intercept", "beta_cdd"]:
        model_warnings.extend(
            get_parameter_negative_warning(model_type, model_params, parameter)
        )
    model_warnings.extend(
        get_parameter_p_value_too_high_warning(
            model_type,
            model_params,
            parameter,
            beta_cdd_p_value,
            beta_cdd_maximum_p_value,
        )
    )

    if len(model_warnings) > 0:
        status = "DISQUALIFIED"
    else:
        status = "QUALIFIED"

    return _candidate_model_factory(
        model_type,
        formula,
        status,
        warnings=model_warnings,
        model_params=model_params,
        model=model,
        result=result,
        r_squared_adj=r_squared_adj,
    )


def get_cdd_only_candidate_models(
    data, minimum_non_zero_cdd, minimum_total_cdd, beta_cdd_maximum_p_value, weights_col
):
    """ Return a list of all possible candidate cdd-only models.

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and 1 to n
        columns with names of the form ``cdd_<balance_point>``. All columns
        with names of this form will be used to fit a candidate model.
        DataFrames of this form can be made using the
        :any:`eemeter.merge_temperature_data` method.
    minimum_non_zero_cdd : :any:`int`
        Minimum allowable number of non-zero cooling degree day values.
    minimum_total_cdd : :any:`float`
        Minimum allowable total sum of cooling degree day values.
    beta_cdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta cdd parameter.
    weights_col : :any:`str` or None
        The name of the column (if any) in ``data`` to use as weights.

    Returns
    -------
    candidate_models : :any:`list` of :any:`CandidateModel`
        A list of cdd-only candidate models, with any associated warnings.
    """
    balance_points = [int(col[4:]) for col in data.columns if col.startswith("cdd")]
    candidate_models = [
        get_single_cdd_only_candidate_model(
            data,
            minimum_non_zero_cdd,
            minimum_total_cdd,
            beta_cdd_maximum_p_value,
            weights_col,
            balance_point,
        )
        for balance_point in balance_points
    ]
    return candidate_models


def get_single_hdd_only_candidate_model(
    data,
    minimum_non_zero_hdd,
    minimum_total_hdd,
    beta_hdd_maximum_p_value,
    weights_col,
    balance_point,
):
    """ Return a single candidate hdd-only model for a particular balance
    point.

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and
        ``hdd_<balance_point>``
        DataFrames of this form can be made using the
        :any:`eemeter.merge_temperature_data` method.
    minimum_non_zero_hdd : :any:`int`
        Minimum allowable number of non-zero heating degree day values.
    minimum_total_hdd : :any:`float`
        Minimum allowable total sum of heating degree day values.
    beta_hdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta hdd parameter.
    weights_col : :any:`str` or None
        The name of the column (if any) in ``data`` to use as weights.
    balance_point : :any:`float`
        The heating balance point for this model.

    Returns
    -------
    candidate_model : :any:`CandidateModel`
        A single hdd-only candidate model, with any associated warnings.
    """
    model_type = "hdd_only"
    hdd_column = "hdd_%s" % balance_point
    formula = "meter_value ~ %s" % hdd_column

    degree_day_warnings = []
    degree_day_warnings.extend(
        get_total_degree_day_too_low_warning(
            model_type, balance_point, "hdd", data[hdd_column], minimum_total_hdd
        )
    )
    degree_day_warnings.extend(
        get_too_few_non_zero_degree_day_warning(
            model_type, balance_point, "hdd", data[hdd_column], minimum_non_zero_hdd
        )
    )

    if len(degree_day_warnings) > 0:
        return _candidate_model_factory(
            model_type,
            formula,
            "NOT ATTEMPTED",
            warnings=degree_day_warnings,
            use_predict_func=False,
        )

    if weights_col is None:
        weights = 1
    else:
        weights = data[weights_col]

    try:
        model = smf.wls(formula=formula, data=data, weights=weights)
    except Exception as e:
        return get_fit_failed_candidate_model(model_type, formula)

    result = model.fit()
    r_squared_adj = result.rsquared_adj
    beta_hdd_p_value = result.pvalues[hdd_column]

    # CalTrack 3.3.1.3
    model_params = {
        "intercept": result.params["Intercept"],
        "beta_hdd": result.params[hdd_column],
        "heating_balance_point": balance_point,
    }

    model_warnings = []

    # CalTrack 3.4.3.2
    for parameter in ["intercept", "beta_hdd"]:
        model_warnings.extend(
            get_parameter_negative_warning(model_type, model_params, parameter)
        )
    model_warnings.extend(
        get_parameter_p_value_too_high_warning(
            model_type,
            model_params,
            parameter,
            beta_hdd_p_value,
            beta_hdd_maximum_p_value,
        )
    )

    if len(model_warnings) > 0:
        status = "DISQUALIFIED"
    else:
        status = "QUALIFIED"

    return _candidate_model_factory(
        model_type,
        formula,
        status,
        warnings=model_warnings,
        model_params=model_params,
        model=model,
        result=result,
        r_squared_adj=r_squared_adj,
    )


def get_hdd_only_candidate_models(
    data, minimum_non_zero_hdd, minimum_total_hdd, beta_hdd_maximum_p_value, weights_col
):
    """
    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and 1 to n
        columns with names of the form ``hdd_<balance_point>``. All columns
        with names of this form will be used to fit a candidate model.
        DataFrames of this form can be made using the
        :any:`eemeter.merge_temperature_data` method.
    minimum_non_zero_hdd : :any:`int`
        Minimum allowable number of non-zero heating degree day values.
    minimum_total_hdd : :any:`float`
        Minimum allowable total sum of heating degree day values.
    beta_hdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta hdd parameter.
    weights_col : :any:`str` or None
        The name of the column (if any) in ``data`` to use as weights.

    Returns
    -------
    candidate_models : :any:`list` of :any:`CandidateModel`
        A list of hdd-only candidate models, with any associated warnings.
    """

    balance_points = [int(col[4:]) for col in data.columns if col.startswith("hdd")]

    candidate_models = [
        get_single_hdd_only_candidate_model(
            data,
            minimum_non_zero_hdd,
            minimum_total_hdd,
            beta_hdd_maximum_p_value,
            weights_col,
            balance_point,
        )
        for balance_point in balance_points
    ]
    return candidate_models


def get_single_cdd_hdd_candidate_model(
    data,
    minimum_non_zero_cdd,
    minimum_non_zero_hdd,
    minimum_total_cdd,
    minimum_total_hdd,
    beta_cdd_maximum_p_value,
    beta_hdd_maximum_p_value,
    weights_col,
    cooling_balance_point,
    heating_balance_point,
):
    """ Return a single candidate cdd_hdd model for a particular selection
    of cooling balance point and heating balance point

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and
        ``hdd_<heating_balance_point>`` and ``cdd_<cooling_balance_point>``
        DataFrames of this form can be made using the
        :any:`eemeter.merge_temperature_data` method.
    minimum_non_zero_cdd : :any:`int`
        Minimum allowable number of non-zero cooling degree day values.
    minimum_non_zero_hdd : :any:`int`
        Minimum allowable number of non-zero heating degree day values.
    minimum_total_cdd : :any:`float`
        Minimum allowable total sum of cooling degree day values.
    minimum_total_hdd : :any:`float`
        Minimum allowable total sum of heating degree day values.
    beta_cdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta cdd parameter.
    beta_hdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta hdd parameter.
    weights_col : :any:`str` or None
        The name of the column (if any) in ``data`` to use as weights.
    cooling_balance_point : :any:`float`
        The cooling balance point for this model.
    heating_balance_point : :any:`float`
        The heating balance point for this model.

    Returns
    -------
    candidate_model : :any:`CandidateModel`
        A single cdd-hdd candidate model, with any associated warnings.
    """
    model_type = "cdd_hdd"
    cdd_column = "cdd_%s" % cooling_balance_point
    hdd_column = "hdd_%s" % heating_balance_point
    formula = "meter_value ~ %s + %s" % (cdd_column, hdd_column)

    degree_day_warnings = []
    degree_day_warnings.extend(
        get_total_degree_day_too_low_warning(
            model_type,
            cooling_balance_point,
            "cdd",
            data[cdd_column],
            minimum_total_cdd,
        )
    )
    degree_day_warnings.extend(
        get_too_few_non_zero_degree_day_warning(
            model_type,
            cooling_balance_point,
            "cdd",
            data[cdd_column],
            minimum_non_zero_cdd,
        )
    )
    degree_day_warnings.extend(
        get_total_degree_day_too_low_warning(
            model_type,
            heating_balance_point,
            "hdd",
            data[hdd_column],
            minimum_total_hdd,
        )
    )
    degree_day_warnings.extend(
        get_too_few_non_zero_degree_day_warning(
            model_type,
            heating_balance_point,
            "hdd",
            data[hdd_column],
            minimum_non_zero_hdd,
        )
    )

    if len(degree_day_warnings) > 0:
        return _candidate_model_factory(
            model_type,
            formula,
            "NOT ATTEMPTED",
            warnings=degree_day_warnings,
            use_predict_func=False,
        )

    if weights_col is None:
        weights = 1
    else:
        weights = data[weights_col]

    try:
        model = smf.wls(formula=formula, data=data, weights=weights)
    except Exception as e:
        return get_fit_failed_candidate_model(model_type, formula)

    result = model.fit()
    r_squared_adj = result.rsquared_adj
    beta_cdd_p_value = result.pvalues[cdd_column]
    beta_hdd_p_value = result.pvalues[hdd_column]

    # CalTrack 3.3.1.3
    model_params = {
        "intercept": result.params["Intercept"],
        "beta_cdd": result.params[cdd_column],
        "beta_hdd": result.params[hdd_column],
        "cooling_balance_point": cooling_balance_point,
        "heating_balance_point": heating_balance_point,
    }

    model_warnings = []

    # CalTrack 3.4.3.2
    for parameter in ["intercept", "beta_cdd", "beta_hdd"]:
        model_warnings.extend(
            get_parameter_negative_warning(model_type, model_params, parameter)
        )
    model_warnings.extend(
        get_parameter_p_value_too_high_warning(
            model_type,
            model_params,
            parameter,
            beta_cdd_p_value,
            beta_cdd_maximum_p_value,
        )
    )
    model_warnings.extend(
        get_parameter_p_value_too_high_warning(
            model_type,
            model_params,
            parameter,
            beta_hdd_p_value,
            beta_hdd_maximum_p_value,
        )
    )

    if len(model_warnings) > 0:
        status = "DISQUALIFIED"
    else:
        status = "QUALIFIED"

    return _candidate_model_factory(
        model_type,
        formula,
        status,
        warnings=model_warnings,
        model_params=model_params,
        model=model,
        result=result,
        r_squared_adj=r_squared_adj,
    )


def get_cdd_hdd_candidate_models(
    data,
    minimum_non_zero_cdd,
    minimum_non_zero_hdd,
    minimum_total_cdd,
    minimum_total_hdd,
    beta_cdd_maximum_p_value,
    beta_hdd_maximum_p_value,
    weights_col,
):
    """ Return a list of candidate cdd_hdd models for a particular selection
    of cooling balance point and heating balance point

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and 1 to n
        columns each of the form ``hdd_<heating_balance_point>``
        and ``cdd_<cooling_balance_point>``. DataFrames of this form can be
        made using the :any:`eemeter.merge_temperature_data` method.
    minimum_non_zero_cdd : :any:`int`
        Minimum allowable number of non-zero cooling degree day values.
    minimum_non_zero_hdd : :any:`int`
        Minimum allowable number of non-zero heating degree day values.
    minimum_total_cdd : :any:`float`
        Minimum allowable total sum of cooling degree day values.
    minimum_total_hdd : :any:`float`
        Minimum allowable total sum of heating degree day values.
    beta_cdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta cdd parameter.
    beta_hdd_maximum_p_value : :any:`float`
        The maximum allowable p-value of the beta hdd parameter.
    weights_col : :any:`str` or None
        The name of the column (if any) in ``data`` to use as weights.

    Returns
    -------
    candidate_models : :any:`list` of :any:`CandidateModel`
        A list of cdd_hdd candidate models, with any associated warnings.
    """

    cooling_balance_points = [
        int(col[4:]) for col in data.columns if col.startswith("cdd")
    ]
    heating_balance_points = [
        int(col[4:]) for col in data.columns if col.startswith("hdd")
    ]

    # CalTrack 3.2.2.1
    candidate_models = [
        get_single_cdd_hdd_candidate_model(
            data,
            minimum_non_zero_cdd,
            minimum_non_zero_hdd,
            minimum_total_cdd,
            minimum_total_hdd,
            beta_cdd_maximum_p_value,
            beta_hdd_maximum_p_value,
            weights_col,
            cooling_balance_point,
            heating_balance_point,
        )
        for cooling_balance_point in cooling_balance_points
        for heating_balance_point in heating_balance_points
        if heating_balance_point <= cooling_balance_point
    ]
    return candidate_models


def select_best_candidate(candidate_models):
    """ Select and return the best candidate model based on r-squared and
    qualification.

    Parameters
    ----------
    candidate_models : :any:`list` of :any:`eemeter.CandidateModel`
        Candidate models to select from.

    Returns
    -------
    (best_candidate, warnings) : :any:`tuple` of :any:`eemeter.CandidateModel` or :any:`None` and :any:`list` of `eemeter.EEMeterWarning`
        Return the candidate model with highest r-squared or None if none meet
        the requirements, and a list of warnings about this selection (or lack
        of selection).
    """
    best_r_squared_adj = -np.inf
    best_candidate = None

    # CalTrack 3.4.3.3
    for candidate in candidate_models:
        if (
            candidate.status == "QUALIFIED"
            and candidate.r_squared_adj > best_r_squared_adj
        ):
            best_candidate = candidate
            best_r_squared_adj = candidate.r_squared_adj

    if best_candidate is None:
        warnings = [
            EEMeterWarning(
                qualified_name="eemeter.caltrack_daily.select_best_candidate.no_candidates",
                description="No qualified model candidates available.",
                data={
                    "status_count:{}".format(status): count
                    for status, count in Counter(
                        [c.status for c in candidate_models]
                    ).items()
                },
            )
        ]
        return None, warnings

    return best_candidate, []


def caltrack_method(
    data,
    fit_cdd=True,
    use_billing_presets=False,
    minimum_non_zero_cdd=10,
    minimum_non_zero_hdd=10,
    minimum_total_cdd=20,
    minimum_total_hdd=20,
    beta_cdd_maximum_p_value=1,
    beta_hdd_maximum_p_value=1,
    weights_col=None,
    fit_intercept_only=True,
    fit_cdd_only=True,
    fit_hdd_only=True,
    fit_cdd_hdd=True,
):
    """ CalTRACK method.

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and 1 to n
        columns each of the form ``hdd_<heating_balance_point>``
        and ``cdd_<cooling_balance_point>``. DataFrames of this form can be
        made using the :any:`eemeter.merge_temperature_data` method.
    fit_cdd : :any:`bool`, optional
        If True, fit CDD models unless overridden by ``fit_cdd_only`` or
        ``fit_cdd_hdd`` flags. Should be set to ``False`` for gas meter data.
    use_billing_presets : :any:`bool`, optional
        Use presets appropriate for billing models. Otherwise defaults are
        appropriate for daily models.
    minimum_non_zero_cdd : :any:`int`, optional
        Minimum allowable number of non-zero cooling degree day values.
    minimum_non_zero_hdd : :any:`int`, optional
        Minimum allowable number of non-zero heating degree day values.
    minimum_total_cdd : :any:`float`, optional
        Minimum allowable total sum of cooling degree day values.
    minimum_total_hdd : :any:`float`, optional
        Minimum allowable total sum of heating degree day values.
    beta_cdd_maximum_p_value : :any:`float`, optional
        The maximum allowable p-value of the beta cdd parameter. The default
        value is the most permissive possible (i.e., 1). This is here
        for backwards compatibility with CalTRACK 1.0 methods.
    beta_hdd_maximum_p_value : :any:`float`, optional
        The maximum allowable p-value of the beta hdd parameter. The default
        value is the most permissive possible (i.e., 1). This is here
        for backwards compatibility with CalTRACK 1.0 methods.
    weights_col : :any:`str` or None, optional
        The name of the column (if any) in ``data`` to use as weights.
    fit_intercept_only : :any:`bool`, optional
        If True, fit and consider intercept_only model candidates.
    fit_cdd_only : :any:`bool`, optional
        If True, fit and consider cdd_only model candidates. Ignored if
        ``fit_cdd=False``.
    fit_hdd_only : :any:`bool`, optional
        If True, fit and consider hdd_only model candidates.
    fit_cdd_hdd : :any:`bool`, optional
        If True, fit and consider cdd_hdd model candidates. Ignored if
        ``fit_cdd=False``.

    Returns
    -------
    model_results : :any:`eemeter.ModelResults`
        Results of running CalTRACK daily method. See :any:`eemeter.ModelResults`
        for more details.
    """
    if use_billing_presets:
        minimum_non_zero_cdd = 0
        minimum_non_zero_hdd = 0
        minimum_total_cdd = 0
        minimum_total_hdd = 0

    # cleans data to fully NaN rows that have missing temp or meter data
    data = overwrite_partial_rows_with_nan(data)

    if data.empty:
        return ModelResults(
            status="NO DATA",
            method_name="caltrack_method",
            warnings=[
                EEMeterWarning(
                    qualified_name="eemeter.caltrack_method.no_data",
                    description=("No data available. Cannot fit model."),
                    data={},
                )
            ],
        )

    # collect all candidate results, then validate all at once
    # CalTrack 3.4.3.1
    candidates = []

    if fit_intercept_only:
        candidates.extend(
            get_intercept_only_candidate_models(data, weights_col=weights_col)
        )

    if fit_hdd_only:
        candidates.extend(
            get_hdd_only_candidate_models(
                data=data,
                minimum_non_zero_hdd=minimum_non_zero_hdd,
                minimum_total_hdd=minimum_total_hdd,
                beta_hdd_maximum_p_value=beta_hdd_maximum_p_value,
                weights_col=weights_col,
            )
        )

    # cdd models ignored for gas
    if fit_cdd:
        if fit_cdd_only:
            candidates.extend(
                get_cdd_only_candidate_models(
                    data=data,
                    minimum_non_zero_cdd=minimum_non_zero_cdd,
                    minimum_total_cdd=minimum_total_cdd,
                    beta_cdd_maximum_p_value=beta_cdd_maximum_p_value,
                    weights_col=weights_col,
                )
            )

        if fit_cdd_hdd:
            candidates.extend(
                get_cdd_hdd_candidate_models(
                    data=data,
                    minimum_non_zero_cdd=minimum_non_zero_cdd,
                    minimum_non_zero_hdd=minimum_non_zero_hdd,
                    minimum_total_cdd=minimum_total_cdd,
                    minimum_total_hdd=minimum_total_hdd,
                    beta_cdd_maximum_p_value=beta_cdd_maximum_p_value,
                    beta_hdd_maximum_p_value=beta_hdd_maximum_p_value,
                    weights_col=weights_col,
                )
            )

    # find best candidate result
    best_candidate, candidate_warnings = select_best_candidate(candidates)

    warnings = candidate_warnings

    if best_candidate is None:
        status = "NO MODEL"
        r_squared_adj = None
    else:
        status = "SUCCESS"
        r_squared_adj = best_candidate.r_squared_adj

    model_result = ModelResults(
        status=status,
        method_name="caltrack_method",
        model=best_candidate,
        candidates=candidates,
        r_squared_adj=r_squared_adj,
        warnings=warnings,
        settings={
            "fit_cdd": fit_cdd,
            "minimum_non_zero_cdd": minimum_non_zero_cdd,
            "minimum_non_zero_hdd": minimum_non_zero_hdd,
            "minimum_total_cdd": minimum_total_cdd,
            "minimum_total_hdd": minimum_total_hdd,
            "beta_cdd_maximum_p_value": beta_cdd_maximum_p_value,
            "beta_hdd_maximum_p_value": beta_hdd_maximum_p_value,
        },
    )

    if best_candidate is not None:
        if best_candidate.model_type in ["cdd_hdd"]:
            num_parameters = 2
        elif best_candidate.model_type in ["hdd_only", "cdd_only"]:
            num_parameters = 1
        else:
            num_parameters = 0

        predicted = _caltrack_predict_design_matrix(
            best_candidate.model_type,
            best_candidate.model_params,
            data,
            input_averages=True,
            output_averages=True,
        )
        model_result.metrics = ModelMetrics(data.meter_value, predicted, num_parameters)

    return model_result


def caltrack_sufficiency_criteria(
    data_quality,
    requested_start,
    requested_end,
    num_days=365,
    min_fraction_daily_coverage=0.9,  # TODO: needs to be per year
    min_fraction_hourly_temperature_coverage_per_period=0.9,
):
    """CalTRACK daily data sufficiency criteria.

    .. note::

        For CalTRACK compliance, ``min_fraction_daily_coverage`` must be set
        at ``0.9`` (section 2.2.1.2), and requested_start and requested_end must
        not be None (section 2.2.4).

    TODO: add warning for outliers (CalTrack 2.3.6)

    Parameters
    ----------
    data_quality : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and the two
        columns ``temperature_null``, containing a count of null hourly
        temperature values for each meter value, and ``temperature_not_null``,
        containing a count of not-null hourly temperature values for each
        meter value. Should have a :any:`pandas.DatetimeIndex`.
    requested_start : :any:`datetime.datetime`, timezone aware (or :any:`None`)
        The desired start of the period, if any, especially if this is
        different from the start of the data. If given, warnings
        are reported on the basis of this start date instead of data start
        date. Must be explicitly set to ``None`` in order to use data start date.
    requested_end : :any:`datetime.datetime`, timezone aware (or :any:`None`)
        The desired end of the period, if any, especially if this is
        different from the end of the data. If given, warnings
        are reported on the basis of this end date instead of data end date.
        Must be explicitly set to ``None`` in order to use data end date.
    num_days : :any:`int`, optional
        Exact number of days allowed in data, including extent given by
        ``requested_start`` or ``requested_end``, if given.
    min_fraction_daily_coverage : :any:, optional
        Minimum fraction of days of data in total data extent for which data
        must be available.
    min_fraction_hourly_temperature_coverage_per_period=0.9,
        Minimum fraction of hours of temperature data coverage in a particular
        period. Anything below this causes the whole period to be considered
        considered missing.

    Returns
    -------
    data_sufficiency : :any:`eemeter.DataSufficiency`
        The an object containing sufficiency status and warnings for this data.
    """
    criteria_name = "caltrack_sufficiency_criteria"

    if data_quality.empty:
        return DataSufficiency(
            status="NO DATA",
            criteria_name=criteria_name,
            warnings=[
                EEMeterWarning(
                    qualified_name="eemeter.caltrack_sufficiency_criteria.no_data",
                    description=("No data available."),
                    data={},
                )
            ],
        )

    data_start = data_quality.index.min().tz_convert("UTC")
    data_end = data_quality.index.max().tz_convert("UTC")
    n_days_data = (data_end - data_start).days

    if requested_start is not None:
        # check for gap at beginning
        requested_start = requested_start.astimezone(pytz.UTC)
        n_days_start_gap = (data_start - requested_start).days
    else:
        n_days_start_gap = 0

    if requested_end is not None:
        # check for gap at end
        requested_end = requested_end.astimezone(pytz.UTC)
        n_days_end_gap = (requested_end - data_end).days
    else:
        n_days_end_gap = 0

    critical_warnings = []

    if n_days_end_gap < 0:
        # CalTRACK 2.2.4
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".extra_data_after_requested_end_date"
                ),
                description=("Extra data found after requested end date."),
                data={
                    "requested_end": requested_end.isoformat(),
                    "data_end": data_end.isoformat(),
                },
            )
        )
        n_days_end_gap = 0

    if n_days_start_gap < 0:
        # CalTRACK 2.2.4
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".extra_data_before_requested_start_date"
                ),
                description=("Extra data found before requested start date."),
                data={
                    "requested_start": requested_start.isoformat(),
                    "data_start": data_start.isoformat(),
                },
            )
        )
        n_days_start_gap = 0

    n_days_total = n_days_data + n_days_start_gap + n_days_end_gap

    n_negative_meter_values = data_quality.meter_value[
        data_quality.meter_value < 0
    ].shape[0]

    if n_negative_meter_values > 0:
        # CalTrack 2.3.5
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria" ".negative_meter_values"
                ),
                description=(
                    "Found negative meter data values, which may indicate presence"
                    " of solar net metering."
                ),
                data={"n_negative_meter_values": n_negative_meter_values},
            )
        )

    # TODO(philngo): detect and report unsorted or repeated values.

    # create masks showing which daily or billing periods meet criteria
    valid_meter_value_rows = data_quality.meter_value.notnull()
    valid_temperature_rows = (
        data_quality.temperature_not_null
        / (data_quality.temperature_not_null + data_quality.temperature_null)
    ) > min_fraction_hourly_temperature_coverage_per_period
    valid_rows = valid_meter_value_rows & valid_temperature_rows

    # get number of days per period - for daily this should be a series of ones
    row_day_counts = day_counts(data_quality.meter_value)

    # apply masks, giving total
    n_valid_meter_value_days = int((valid_meter_value_rows * row_day_counts).sum())
    n_valid_temperature_days = int((valid_temperature_rows * row_day_counts).sum())
    n_valid_days = int((valid_rows * row_day_counts).sum())

    if n_days_total > 0:
        fraction_valid_meter_value_days = n_valid_meter_value_days / float(n_days_total)
        fraction_valid_temperature_days = n_valid_temperature_days / float(n_days_total)
        fraction_valid_days = n_valid_days / float(n_days_total)
    else:
        # unreachable, I think.
        fraction_valid_meter_value_days = 0
        fraction_valid_temperature_days = 0
        fraction_valid_days = 0

    if n_days_total != num_days:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".incorrect_number_of_total_days"
                ),
                description=("Total data span does not match the required value."),
                data={"num_days": num_days, "n_days_total": n_days_total},
            )
        )

    if fraction_valid_days < min_fraction_daily_coverage:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".too_many_days_with_missing_data"
                ),
                description=(
                    "Too many days in data have missing meter data or"
                    " temperature data."
                ),
                data={"n_valid_days": n_valid_days, "n_days_total": n_days_total},
            )
        )

    if fraction_valid_meter_value_days < min_fraction_daily_coverage:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".too_many_days_with_missing_meter_data"
                ),
                description=("Too many days in data have missing meter data."),
                data={
                    "n_valid_meter_data_days": n_valid_meter_value_days,
                    "n_days_total": n_days_total,
                },
            )
        )

    if fraction_valid_temperature_days < min_fraction_daily_coverage:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".too_many_days_with_missing_temperature_data"
                ),
                description=("Too many days in data have missing temperature data."),
                data={
                    "n_valid_temperature_data_days": n_valid_temperature_days,
                    "n_days_total": n_days_total,
                },
            )
        )

    if len(critical_warnings) > 0:
        status = "FAIL"
    else:
        status = "PASS"

    warnings = critical_warnings

    return DataSufficiency(
        status=status,
        criteria_name=criteria_name,
        warnings=warnings,
        settings={
            "num_days": num_days,
            "min_fraction_daily_coverage": min_fraction_daily_coverage,
            "min_fraction_hourly_temperature_coverage_per_period": min_fraction_hourly_temperature_coverage_per_period,
        },
    )


def caltrack_metered_savings(
    baseline_model,
    reporting_meter_data,
    temperature_data,
    degree_day_method="daily",
    with_disaggregated=False,
):
    """ Compute metered savings, i.e., savings in which the baseline model
    is used to calculate the modeled usage in the reporting period. This
    modeled usage is then compared to the actual usage from the reporting period.

    Parameters
    ----------
    baseline_model : :any:`eemeter.CandidateModel`
        Model to use for predicting pre-intervention usage.
    reporting_meter_data : :any:`pandas.DataFrame`
        The observed reporting period data. Savings will be computed for the
        periods supplied in the reporting period data.
    temperature_data : :any:`pandas.Series`
        Hourly-frequency timeseries of temperature data during the reporting
        period.
    degree_day_method : :any:`str`, optional
        The method to use to calculate degree days using hourly temperature
        data. Can be either ``'hourly'`` or ``'daily'``.
    with_disaggregated : :any:`bool`, optional
        If True, calculate baseline counterfactual disaggregated usage
        estimates. Savings cannot be disaggregated for metered savings. For
        that, use :any:`eemeter.caltrack_modeled_savings`.

    Returns
    -------
    results : :any:`pandas.DataFrame`
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
    prediction_index = reporting_meter_data.index
    predicted_baseline_usage = baseline_model.predict(
        temperature_data, prediction_index, degree_day_method, with_disaggregated=True
    )
    # CalTrack 3.5.1
    counterfactual_usage = predicted_baseline_usage["predicted_usage"].to_frame(
        "counterfactual_usage"
    )

    reporting_observed = reporting_meter_data["value"].to_frame("reporting_observed")

    def metered_savings_func(row):
        return row.counterfactual_usage - row.reporting_observed

    results = reporting_observed.join(counterfactual_usage).assign(
        metered_savings=metered_savings_func
    )

    if with_disaggregated:
        counterfactual_usage_disaggregated = predicted_baseline_usage[
            ["base_load", "heating_load", "cooling_load"]
        ].rename(
            columns={
                "base_load": "counterfactual_base_load",
                "heating_load": "counterfactual_heating_load",
                "cooling_load": "counterfactual_cooling_load",
            }
        )
        results = results.join(counterfactual_usage_disaggregated)

    return results.dropna().reindex(results.index)


def caltrack_modeled_savings(
    baseline_model,
    reporting_model,
    result_index,
    temperature_data,
    degree_day_method="daily",
    with_disaggregated=False,
):
    """ Compute modeled savings, i.e., savings in which baseline and reporting
    usage values are based on models. This is appropriate for annualizing or
    weather normalizing models.

    Parameters
    ----------
    baseline_model : :any:`eemeter.CandidateModel`
        Model to use for predicting pre-intervention usage.
    reporting_model : :any:`eemeter.CandidateModel`
        Model to use for predicting post-intervention usage.
    result_index : :any:`pandas.DatetimeIndex`
        The dates for which usage should be modeled.
    temperature_data : :any:`pandas.Series`
        Hourly-frequency timeseries of temperature data during the modeled
        period.
    degree_day_method : :any:`str`, optional
        The method to use to calculate degree days using hourly temperature
        data. Can be either ``'hourly'`` or ``'daily'``.
    with_disaggregated : :any:`bool`, optional
        If True, calculate modeled disaggregated usage estimates and savings.

    Returns
    -------
    results : :any:`pandas.DataFrame`
        DataFrame with modeled savings, indexed with the result_index. Will
        include the following columns:

        - ``modeled_baseline_usage``
        - ``modeled_reporting_usage``
        - ``modeled_savings``

        If `with_disaggregated` is set to True, the following columns will also
        be in the results DataFrame:

        - ``modeled_baseline_base_load``
        - ``modeled_baseline_cooling_load``
        - ``modeled_baseline_heating_load``
        - ``modeled_reporting_base_load``
        - ``modeled_reporting_cooling_load``
        - ``modeled_reporting_heating_load``
        - ``modeled_base_load_savings``
        - ``modeled_cooling_load_savings``
        - ``modeled_heating_load_savings``
    """
    prediction_index = result_index

    predicted_baseline_usage = baseline_model.predict(
        temperature_data, prediction_index, degree_day_method, with_disaggregated=True
    )
    modeled_baseline_usage = predicted_baseline_usage["predicted_usage"].to_frame(
        "modeled_baseline_usage"
    )

    predicted_reporting_usage = reporting_model.predict(
        temperature_data, prediction_index, degree_day_method, with_disaggregated=True
    )
    modeled_reporting_usage = predicted_reporting_usage["predicted_usage"].to_frame(
        "modeled_reporting_usage"
    )

    def modeled_savings_func(row):
        return row.modeled_baseline_usage - row.modeled_reporting_usage

    results = modeled_baseline_usage.join(modeled_reporting_usage).assign(
        modeled_savings=modeled_savings_func
    )

    if with_disaggregated:
        modeled_baseline_usage_disaggregated = predicted_baseline_usage[
            ["base_load", "heating_load", "cooling_load"]
        ].rename(
            columns={
                "base_load": "modeled_baseline_base_load",
                "heating_load": "modeled_baseline_heating_load",
                "cooling_load": "modeled_baseline_cooling_load",
            }
        )

        modeled_reporting_usage_disaggregated = predicted_reporting_usage[
            ["base_load", "heating_load", "cooling_load"]
        ].rename(
            columns={
                "base_load": "modeled_reporting_base_load",
                "heating_load": "modeled_reporting_heating_load",
                "cooling_load": "modeled_reporting_cooling_load",
            }
        )

        def modeled_base_load_savings_func(row):
            return row.modeled_baseline_base_load - row.modeled_reporting_base_load

        def modeled_heating_load_savings_func(row):
            return (
                row.modeled_baseline_heating_load - row.modeled_reporting_heating_load
            )

        def modeled_cooling_load_savings_func(row):
            return (
                row.modeled_baseline_cooling_load - row.modeled_reporting_cooling_load
            )

        results = (
            results.join(modeled_baseline_usage_disaggregated)
            .join(modeled_reporting_usage_disaggregated)
            .assign(
                modeled_base_load_savings=modeled_base_load_savings_func,
                modeled_heating_load_savings=modeled_heating_load_savings_func,
                modeled_cooling_load_savings=modeled_cooling_load_savings_func,
            )
        )

    return results.dropna().reindex(results.index)


def plot_caltrack_candidate(
    candidate,
    best=False,
    ax=None,
    title=None,
    figsize=None,
    temp_range=None,
    alpha=None,
    **kwargs
):
    """ Plot a CalTRACK candidate model.

    Parameters
    ----------
    candidate : :any:`eemeter.CandidateModel`
        A candidate model with a predict function.
    best : :any:`bool`, optional
        Whether this is the best candidate or not.
    ax : :any:`matplotlib.axes.Axes`, optional
        Existing axes to plot on.
    title : :any:`str`, optional
        Chart title.
    figsize : :any:`tuple`, optional
        (width, height) of chart.
    temp_range : :any:`tuple`, optional
        (min, max) temperatures to plot model.
    alpha : :any:`float` between 0 and 1, optional
        Transparency, 0 fully transparent, 1 fully opaque.
    **kwargs
        Keyword arguments for :any:`matplotlib.axes.Axes.plot`

    Returns
    -------
    ax : :any:`matplotlib.axes.Axes`
        Matplotlib axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.")

    if figsize is None:
        figsize = (10, 4)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if candidate.status == "QUALIFIED":
        color = "C2"
    elif candidate.status == "DISQUALIFIED":
        color = "C3"
    else:
        return

    if best:
        color = "C1"
        alpha = 1

    temp_min, temp_max = (30, 90) if temp_range is None else temp_range

    temps = np.arange(temp_min, temp_max)

    data = {"n_days": np.ones(temps.shape)}

    prediction_index = pd.date_range(
        "2017-01-01T00:00:00Z", periods=len(temps), freq="D"
    )

    temps_hourly = pd.Series(temps, index=prediction_index).resample("H").ffill()

    prediction = candidate.predict(temps_hourly, prediction_index, "daily")

    plot_kwargs = {"color": color, "alpha": alpha or 0.3}
    plot_kwargs.update(kwargs)

    ax.plot(temps, prediction, **plot_kwargs)

    if title is not None:
        ax.set_title(title)

    return ax
