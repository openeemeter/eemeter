__all__ = ("metered_savings", "modeled_savings")


def _compute_ols_error(
    t_stat,
    rmse_base_residuals,
    post_obs,
    base_obs,
    base_avg,
    post_avg,
    base_var,
    nprime,
):
    ols_model_agg_error = (
        (t_stat * rmse_base_residuals * post_obs)
        / (base_obs ** 0.5)
        * (1 + ((base_avg - post_avg) ** 2 / base_var)) ** 0.5
    )

    ols_noise_agg_error = (
        t_stat * rmse_base_residuals * (post_obs * base_obs / nprime) ** 0.5
    )

    ols_total_agg_error = (ols_model_agg_error ** 2 + ols_noise_agg_error ** 2) ** 0.5

    return ols_total_agg_error, ols_model_agg_error, ols_noise_agg_error


def _compute_fsu_error(
    t_stat,
    frequency,
    post_obs,
    total_base_energy,
    rmse_base_residuals,
    base_avg,
    base_obs,
    nprime,
):
    if frequency == "billing":
        a_coeff = -0.00022
        b_coeff = 0.03306
        c_coeff = 0.94054
        months_reporting = post_obs
    else:
        a_coeff = -0.00024
        b_coeff = 0.03535
        c_coeff = 1.00286
        months_reporting = post_obs / 30

    fsu_error_band = total_base_energy * (
        t_stat
        * (a_coeff * months_reporting ** 2 + b_coeff * months_reporting + c_coeff)
        * (rmse_base_residuals / base_avg)
        * ((base_obs / nprime) * (1 + (2 / nprime)) * (1 / post_obs)) ** 0.5
    )

    return fsu_error_band


def metered_savings(
    baseline_model_results,
    reporting_meter_data,
    temperature_data,
    degree_day_method="daily",
    with_disaggregated=False,
    frequency="unknown",
    t_stat=1.649,
):
    """ Compute metered savings, i.e., savings in which the baseline model
    is used to calculate the modeled usage in the reporting period. This
    modeled usage is then compared to the actual usage from the reporting period.
    Also compute two measures of the uncertainty of the aggregate savings estimate,
    a fractional savings uncertainty (FSU) error band and an OLS error band. (To convert
    the FSU error band into FSU, divide by total estimated savings.)

    Parameters
    ----------
    baseline_model_results : :any:`eemeter.ModelResults`
        ModelResult object to use for predicting pre-intervention usage.
    reporting_meter_data : :any:`pandas.DataFrame`
        The observed reporting period data (totals). Savings will be computed for the
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
    frequency : :any`str`, optional
        The frequency used for calculating the FSU error band. Frequency can be
        ``'billing'`` or ``'daily'``. If frequency is anything else, the FSU and OLS
        error bands will not be calculated.
    t_stat : :any`float`
        The t-statistic associated with the desired confidence level and degrees of
        freedom (number of baseline observations minus the number of parameters).
        Defaults to 1.649, the t-statistic associated with 363 degrees of freedom (365
        observations minus two parameters) and a two-tailed 90% confidence level.

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

    error_bands : :any:`dict`, optional
        If frequency is 'daily' or 'billing', will also return a dictionary of FSU and
        OLS error bands for the aggregated energy savings over the post period.
    """
    prediction_index = reporting_meter_data.index
    model_prediction = baseline_model_results.model.predict(
        temperature_data, prediction_index, degree_day_method, with_disaggregated=True
    )

    predicted_baseline_usage = model_prediction.result

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

    error_bands = None

    if frequency == "daily" or frequency == "billing":
        num_parameters = baseline_model_results.totals_metrics.num_parameters

        base_obs = baseline_model_results.totals_metrics.observed_length
        post_obs = results["reporting_observed"].dropna().shape[0]

        rmse_base_residuals = baseline_model_results.totals_metrics.rmse_adj
        autocorr_resid = baseline_model_results.totals_metrics.autocorr_resid

        base_avg = baseline_model_results.totals_metrics.observed_mean
        post_avg = results["reporting_observed"].mean()
        post_prediction_avg = results["counterfactual_usage"].mean()

        base_var = baseline_model_results.totals_metrics.observed_variance

        nprime = base_obs * (1 - autocorr_resid) / (1 + autocorr_resid)

        total_base_energy = base_avg * base_obs

        ols_total_agg_error, ols_model_agg_error, ols_noise_agg_error = _compute_ols_error(
            t_stat,
            rmse_base_residuals,
            post_obs,
            base_obs,
            base_avg,
            post_avg,
            base_var,
            nprime,
        )

        fsu_error_band = _compute_fsu_error(
            t_stat,
            frequency,
            post_obs,
            total_base_energy,
            rmse_base_residuals,
            base_avg,
            base_obs,
            nprime,
        )

        error_bands = {
            "FSU Error Band": fsu_error_band,
            "OLS Error Band": ols_total_agg_error,
            "OLS Error Band: Model Error": ols_model_agg_error,
            "OLS Error Band: Noise": ols_noise_agg_error,
        }

    return results.dropna().reindex(results.index), error_bands


def modeled_savings(
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

    def _predicted_usage(model):
        model_prediction = model.predict(
            temperature_data,
            prediction_index,
            degree_day_method,
            with_disaggregated=True,
        )
        predicted_usage = model_prediction.result
        return predicted_usage

    predicted_baseline_usage = _predicted_usage(baseline_model)
    predicted_reporting_usage = _predicted_usage(reporting_model)
    modeled_baseline_usage = predicted_baseline_usage["predicted_usage"].to_frame(
        "modeled_baseline_usage"
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
