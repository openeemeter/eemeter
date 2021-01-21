#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2019 OpenEEmeter contributors

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
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..features import (
    compute_time_features,
    compute_temperature_bin_features,
    compute_occupancy_feature,
    merge_features,
)
from ..metrics import ModelMetrics
from ..segmentation import CalTRACKSegmentModel, SegmentedModel, fit_model_segments
from ..warnings import EEMeterWarning


__all__ = (
    "CalTRACKHourlyModelResults",
    "CalTRACKHourlyModel",
    "caltrack_hourly_fit_feature_processor",
    "caltrack_hourly_prediction_feature_processor",
    "fit_caltrack_hourly_model_segment",
    "fit_caltrack_hourly_model",
)


class CalTRACKHourlyModelResults(object):
    """Contains information about the chosen model.

    Attributes
    ----------
    status : :any:`str`
        A string indicating the status of this result. Possible statuses:

        - ``'NO DATA'``: No baseline data was available.
        - ``'NO MODEL'``: A complete model could not be constructed.
        - ``'SUCCESS'``: A model was constructed.
    method_name : :any:`str`
        The name of the method used to fit the baseline model.
    model : :any:`eemeter.CalTRACKHourlyModel` or :any:`None`
        The selected model, if any.
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        A list of any warnings reported during the model selection and fitting
        process.
    metadata : :any:`dict`
        An arbitrary dictionary of metadata to be associated with this result.
        This can be used, for example, to tag the results with attributes like
        an ID::

            {
                'id': 'METER_12345678',
            }

    settings : :any:`dict`
        A dictionary of settings used by the method.
    totals_metrics : :any:`ModelMetrics`
        A ModelMetrics object, if one is calculated and associated with this
        model. (This initializes to None.) The ModelMetrics object contains
        model fit information and descriptive statistics about the underlying data,
        with that data expressed as period totals.
    avgs_metrics : :any:`ModelMetrics`
        A ModelMetrics object, if one is calculated and associated with this
        model. (This initializes to None.) The ModelMetrics object contains
        model fit information and descriptive statistics about the underlying data,
        with that data expressed as daily averages.
    """

    def __init__(
        self, status, method_name, model=None, warnings=[], metadata=None, settings=None
    ):
        self.status = status
        self.method_name = method_name

        self.model = model

        self.warnings = warnings

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        if settings is None:
            settings = {}
        self.settings = settings

        self.totals_metrics = None
        self.avgs_metrics = None

    def __repr__(self):
        return "CalTRACKHourlyModelResults(status='{}', method_name='{}')".format(
            self.status, self.method_name
        )

    def json(self, with_candidates=False):
        """Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """

        def _json_or_none(obj):
            return None if obj is None else obj.json()

        def _json_or_none_in_dict(obj):
            return (
                None
                if obj is None
                else {key: _json_or_none(val) for key, val in obj.items()}
            )

        data = {
            "status": self.status,
            "method_name": self.method_name,
            "model": _json_or_none(self.model),
            "warnings": [w.json() for w in self.warnings],
            "metadata": self.metadata,
            "settings": self.settings,
            "totals_metrics": _json_or_none_in_dict(self.totals_metrics),
            "avgs_metrics": _json_or_none_in_dict(self.avgs_metrics),
        }
        return data

    @classmethod
    def from_json(cls, data):
        """Loads a JSON-serializable representation into the model state.

        The input of this function is a dict which can be the result
        of :any:`json.loads`.
        """

        # "model" is a CalTRACKHourlyModel that was serialized
        model = None
        d = data.get("model")
        if d:
            model = CalTRACKHourlyModel.from_json(d)

        c = cls(
            data.get("status"),
            data.get("method_name"),
            model=model,
            warnings=data.get("warnings"),
            metadata=data.get("metadata"),
            settings=data.get("settings"),
        )

        # Note the metrics do not contain all the data needed
        # for reconstruction (like the input pandas) ...
        d = data.get("avgs_metrics")
        if d:
            c.avgs_metrics = ModelMetrics.from_json(d)  # pragma: no cover
        d = data.get("totals_metrics")
        if d:
            c.totals_metrics = ModelMetrics.from_json(d)
        return c

    def predict(self, prediction_index, temperature_data, **kwargs):
        """Predict over a particular index using temperature data.

        Parameters
        ----------
        prediction_index : :any:`pandas.DatetimeIndex`
            Time period over which to predict.
        temperature_data : :any:`pandas.DataFrame`
            Hourly temperature data to use for prediction. Time period should match
            the ``prediction_index`` argument.
        **kwargs
            Extra keyword arguments to send to self.model.predict

        Returns
        -------
        prediction : :any:`pandas.DataFrame`
            The predicted usage values.
        """
        return self.model.predict(prediction_index, temperature_data, **kwargs)


class CalTRACKHourlyModel(SegmentedModel):
    """An object which holds CalTRACK Hourly model data and metadata, and
    which can be used for prediction.

    Attributes
    ----------
    segment_models : :any:`dict` of `eemeter.CalTRACKSegmentModel`
        Dictionary of models for each segment, keys are segment names.
    occupancy_lookup : :any:`pandas.DataFrame`
        A dataframe with occupancy flags for each hour of the week and each segment.
        Segment names are columns, occupancy flags are 0 or 1.
    temperature_bins : :any:`pandas.DataFrame`
        A dataframe of bin endpoint flags for each segment. Segment names are columns.
    """

    def __init__(self, segment_models, occupancy_lookup, temperature_bins):

        self.occupancy_lookup = occupancy_lookup
        self.temperature_bins = temperature_bins
        super(CalTRACKHourlyModel, self).__init__(
            segment_models=segment_models,
            prediction_segment_type="one_month",
            prediction_segment_name_mapping={
                "jan": "dec-jan-feb-weighted",
                "feb": "jan-feb-mar-weighted",
                "mar": "feb-mar-apr-weighted",
                "apr": "mar-apr-may-weighted",
                "may": "apr-may-jun-weighted",
                "jun": "may-jun-jul-weighted",
                "jul": "jun-jul-aug-weighted",
                "aug": "jul-aug-sep-weighted",
                "sep": "aug-sep-oct-weighted",
                "oct": "sep-oct-nov-weighted",
                "nov": "oct-nov-dec-weighted",
                "dec": "nov-dec-jan-weighted",
            },
            prediction_feature_processor=caltrack_hourly_prediction_feature_processor,
            prediction_feature_processor_kwargs={
                "occupancy_lookup": self.occupancy_lookup,
                "temperature_bins": self.temperature_bins,
            },
        )

    def json(self):
        """Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        data = super(CalTRACKHourlyModel, self).json()
        data.update(
            {
                "occupancy_lookup": self.occupancy_lookup.to_json(orient="split"),
                "temperature_bins": self.temperature_bins.to_json(orient="split"),
            }
        )
        return data

    @classmethod
    def from_json(cls, data):
        """Loads a JSON-serializable representation into the model state.

        The input of this function is a dict which can be the result
        of :any:`json.loads`.
        """

        segment_models = [
            CalTRACKSegmentModel.from_json(s) for s in data.get("segment_models")
        ]

        occupancy_lookup = pd.read_json(data.get("occupancy_lookup"), orient="split")
        occupancy_lookup.index = occupancy_lookup.index.astype("category")

        c = cls(
            segment_models,
            occupancy_lookup,
            pd.read_json(data.get("temperature_bins"), orient="split"),
        )

        return c


def caltrack_hourly_fit_feature_processor(
    segment_name, segmented_data, occupancy_lookup, temperature_bins
):
    """A function that takes in temperature data and returns a dataframe of
    features suitable for use with :any:`eemeter.fit_caltrack_hourly_model_segment`.
    Designed for use with :any:`eemeter.iterate_segmented_dataset`.

    Parameters
    ----------
    segment_name : :any:`str`
        The name of the segment.
    segmented_data : :any:`pandas.DataFrame`
        Hourly temperature data for the segment.
    occupancy_lookup : :any:`pandas.DataFrame`
        A dataframe with occupancy flags for each hour of the week and each segment.
        Segment names are columns, occupancy flags are 0 or 1.
    temperature_bins : :any:`pandas.DataFrame`
        A dataframe of bin endpoint flags for each segment. Segment names are columns.

    Returns
    -------
    features : :any:`pandas.DataFrame`
        A dataframe of features with the following columns:
        - 'meter_value': the observed meter value
        - 'hour_of_week': 0-167
        - 'bin_<0-6>_occupied': temp bin feature, or 0 if unoccupied
        - 'bin_<0-6>_unoccupied': temp bin feature or 0 in occupied
        - 'weight': 0.0 or 0.5 or 1.0
    """
    # get occupied feature
    hour_of_week = segmented_data.hour_of_week
    occupancy = occupancy_lookup[segment_name]
    occupancy_feature = compute_occupancy_feature(hour_of_week, occupancy)

    # get temperature bin features
    temperatures = segmented_data.temperature_mean
    bin_endpoints_list = (
        temperature_bins[segment_name].index[temperature_bins[segment_name]].tolist()
    )
    temperature_bin_features = compute_temperature_bin_features(
        segmented_data.temperature_mean, bin_endpoints_list
    )

    # combine features
    return merge_features(
        [
            segmented_data[["meter_value", "hour_of_week"]],
            occupancy_feature,
            temperature_bin_features,
            segmented_data.weight,
        ]
    )


def caltrack_hourly_prediction_feature_processor(
    segment_name, segmented_data, occupancy_lookup, temperature_bins
):
    """A function that takes in temperature data and returns a dataframe of
    features suitable for use inside :any:`eemeter.CalTRACKHourlyModel`.
    Designed for use with :any:`eemeter.iterate_segmented_dataset`.

    Parameters
    ----------
    segment_name : :any:`str`
        The name of the segment.
    segmented_data : :any:`pandas.DataFrame`
        Hourly temperature data for the segment.
    occupancy_lookup : :any:`pandas.DataFrame`
        A dataframe with occupancy flags for each hour of the week and each segment.
        Segment names are columns, occupancy flags are 0 or 1.
    temperature_bins : :any:`pandas.DataFrame`
        A dataframe of bin endpoint flags for each segment. Segment names are columns.

    Returns
    -------
    features : :any:`pandas.DataFrame`
        A dataframe of features with the following columns:
        - 'hour_of_week': 0-167
        - 'bin_<0-6>_occupied': temp bin feature, or 0 if unoccupied
        - 'bin_<0-6>_unoccupied': temp bin feature or 0 in occupied
        - 'weight': 1
    """
    # hour of week feature
    hour_of_week_feature = compute_time_features(
        segmented_data.index, hour_of_week=True, day_of_week=False, hour_of_day=False
    )

    # occupancy feature
    occupancy = occupancy_lookup[segment_name]
    occupancy_feature = compute_occupancy_feature(
        hour_of_week_feature.hour_of_week, occupancy
    )

    # get temperature bin features
    temperatures = segmented_data
    bin_endpoints_list = (
        temperature_bins[segment_name].index[temperature_bins[segment_name]].tolist()
    )
    temperature_bin_features = compute_temperature_bin_features(
        segmented_data.temperature_mean, bin_endpoints_list
    )

    # combine features
    return merge_features(
        [
            hour_of_week_feature,
            occupancy_feature,
            temperature_bin_features,
            segmented_data.weight,
        ]
    )


def fit_caltrack_hourly_model_segment(segment_name, segment_data):
    """Fit a model for a single segment.

    Parameters
    ----------
    segment_name : :any:`str`
        The name of the segment.
    segment_data : :any:`pandas.DataFrame`
        A design matrix for caltrack hourly, of the form returned by
        :any:`eemeter.caltrack_hourly_prediction_feature_processor`.

    Returns
    -------
    segment_model : :any:`CalTRACKSegmentModel`
        A model that represents the fitted model.
    """

    def _get_hourly_model_formula(data):
        if (np.sum(data.loc[data.weight > 0].occupancy) == 0) or (
            np.sum(data.loc[data.weight > 0].occupancy)
            == len(data.loc[data.weight > 0].occupancy)
        ):
            bin_occupancy_interactions = "".join(
                [" + {}".format(c) for c in data.columns if "bin" in c]
            )
            return "meter_value ~ C(hour_of_week) - 1{}".format(
                bin_occupancy_interactions
            )
        else:
            bin_occupancy_interactions = "".join(
                [" + {}:C(occupancy)".format(c) for c in data.columns if "bin" in c]
            )
            return "meter_value ~ C(hour_of_week) - 1{}".format(
                bin_occupancy_interactions
            )

    warnings = []
    if segment_data.dropna().empty:
        model = None
        formula = None
        model_params = None
        warnings.append(
            EEMeterWarning(
                qualified_name="eemeter.fit_caltrack_hourly_model_segment.no_nonnull_data",
                description="The segment contains either an empty dataset or all NaNs.",
                data={
                    "n_rows": segment_data.shape[0],
                    "n_rows_after_dropna": segment_data.dropna().shape[0],
                },
            )
        )
    else:

        formula = _get_hourly_model_formula(segment_data)
        model = smf.wls(formula=formula, data=segment_data, weights=segment_data.weight)
        model_params = {coeff: value for coeff, value in model.fit().params.items()}

    segment_model = CalTRACKSegmentModel(
        segment_name=segment_name,
        model=model,
        formula=formula,
        model_params=model_params,
        warnings=warnings,
    )
    if model:
        this_segment_data = segment_data[segment_data.weight == 1]
        predicted_value = pd.Series(model.fit().predict(this_segment_data))
        segment_model.totals_metrics = ModelMetrics(
            this_segment_data.meter_value, predicted_value, len(model_params)
        )
    else:
        segment_model.totals_metrics = None

    return segment_model


def fit_caltrack_hourly_model(
    segmented_design_matrices, occupancy_lookup, temperature_bins
):
    """Fit a CalTRACK hourly model

    Parameters
    ----------
    segmented_design_matrices : :any:`dict` of :any:`pandas.DataFrame`
        A dictionary of dataframes of the form returned by
        :any:`eemeter.create_caltrack_hourly_segmented_design_matrices`
    occupancy_lookup : :any:`pandas.DataFrame`
        A dataframe with occupancy flags for each hour of the week and each segment.
        Segment names are columns, occupancy flags are 0 or 1.
    temperature_bins : :any:`pandas.DataFrame`
        A dataframe of bin endpoint flags for each segment. Segment names are columns.

    Returns
    -------
    model : :any:`CalTRACKHourlyModelResults`
        Has a `model.predict` method which take input data and makes a prediction
        using this model.
    """
    segment_models = fit_model_segments(
        segmented_design_matrices, fit_caltrack_hourly_model_segment
    )
    all_warnings = [
        warning
        for segment_model in segment_models
        for warning in segment_model.warnings
    ]
    model = CalTRACKHourlyModel(segment_models, occupancy_lookup, temperature_bins)

    model_results = CalTRACKHourlyModelResults(
        status="SUCCEEDED",
        method_name="caltrack_hourly",
        warnings=all_warnings,
        model=model,
    )
    model_results.totals_metrics = {
        seg_model.segment_name: seg_model.totals_metrics for seg_model in segment_models
    }
    return model_results
