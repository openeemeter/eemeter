__all__ = (
    'metered_savings',
    'modeled_savings',
)

def metered_savings(
    baseline_model, reporting_meter_data, temperature_data,
    degree_day_method='daily', with_disaggregated=False,
):
    ''' Compute metered savings, i.e., savings in which the baseline model
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

    '''
    prediction_index = reporting_meter_data.index
    # with_disaggregated not set in caltrack_hourly_predict
    model_prediction = baseline_model.predict(
        temperature_data, prediction_index, degree_day_method,
        with_disaggregated=True
    )

    predicted_baseline_usage = model_prediction.result
    # CalTrack 3.5.1
    counterfactual_usage = predicted_baseline_usage['predicted_usage']\
        .to_frame('counterfactual_usage')

    reporting_observed = reporting_meter_data['value'].to_frame('reporting_observed')

    def metered_savings_func(row):
        return row.counterfactual_usage - row.reporting_observed

    results = reporting_observed \
        .join(counterfactual_usage) \
        .assign(metered_savings=metered_savings_func)

    if with_disaggregated:
        counterfactual_usage_disaggregated = predicted_baseline_usage[
            ['base_load', 'heating_load', 'cooling_load']
        ].rename(columns={
            'base_load': 'counterfactual_base_load',
            'heating_load': 'counterfactual_heating_load',
            'cooling_load': 'counterfactual_cooling_load',
        })
        results = results.join(counterfactual_usage_disaggregated)

    return results.dropna().reindex(results.index)


def modeled_savings(
    baseline_model, reporting_model, result_index, temperature_data,
    degree_day_method='daily', with_disaggregated=False,
):
    ''' Compute modeled savings, i.e., savings in which baseline and reporting
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
    '''
    prediction_index = result_index

    baseline_model_prediction = baseline_model.predict(temperature_data, prediction_index, degree_day_method,
        with_disaggregated=True
    )
    predicted_baseline_usage = baseline_model_prediction.result
    modeled_baseline_usage = predicted_baseline_usage['predicted_usage']\
        .to_frame('modeled_baseline_usage')

    reporting_model_prediction = reporting_model.predict(
        temperature_data, prediction_index, degree_day_method,
        with_disaggregated=True
    )
    predicted_reporting_usage = reporting_model_prediction.result
    modeled_reporting_usage = predicted_reporting_usage['predicted_usage']\
        .to_frame('modeled_reporting_usage')

    def modeled_savings_func(row):
        return row.modeled_baseline_usage - row.modeled_reporting_usage

    results = modeled_baseline_usage \
        .join(modeled_reporting_usage) \
        .assign(modeled_savings=modeled_savings_func)

    if with_disaggregated:
        modeled_baseline_usage_disaggregated = predicted_baseline_usage[
            ['base_load', 'heating_load', 'cooling_load']
        ].rename(columns={
            'base_load': 'modeled_baseline_base_load',
            'heating_load': 'modeled_baseline_heating_load',
            'cooling_load': 'modeled_baseline_cooling_load',
        })

        modeled_reporting_usage_disaggregated = predicted_reporting_usage[
            ['base_load', 'heating_load', 'cooling_load']
        ].rename(columns={
            'base_load': 'modeled_reporting_base_load',
            'heating_load': 'modeled_reporting_heating_load',
            'cooling_load': 'modeled_reporting_cooling_load',
        })

        def modeled_base_load_savings_func(row):
            return row.modeled_baseline_base_load - row.modeled_reporting_base_load
        def modeled_heating_load_savings_func(row):
            return row.modeled_baseline_heating_load - row.modeled_reporting_heating_load
        def modeled_cooling_load_savings_func(row):
            return row.modeled_baseline_cooling_load - row.modeled_reporting_cooling_load

        results = results.join(modeled_baseline_usage_disaggregated) \
            .join(modeled_reporting_usage_disaggregated) \
            .assign(
                modeled_base_load_savings=modeled_base_load_savings_func,
                modeled_heating_load_savings=modeled_heating_load_savings_func,
                modeled_cooling_load_savings=modeled_cooling_load_savings_func,
            )

    return results.dropna().reindex(results.index)
