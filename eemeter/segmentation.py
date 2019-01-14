#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2018 Open Energy Efficiency, Inc.

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
from collections import namedtuple

import numpy as np
import pandas as pd
from patsy import dmatrix


__all__ = (
    "iterate_segmented_dataset",
    "segment_time_series",
    "CalTRACKSegmentModel",
    "SegmentedModel",
    "HourlyModelPrediction",
)


HourlyModelPrediction = namedtuple("HourlyModelPrediction", ["result"])


class CalTRACKSegmentModel(object):
    def __init__(self, segment_name, model, formula, model_params, warnings=None):
        self.segment_name = segment_name
        self.model = model
        self.formula = formula
        self.model_params = model_params

        if warnings is None:
            warnings = []
        self.warnings = warnings

    def predict(self, data):
        design_matrix_granular = dmatrix(
            self.formula.split("~", 1)[1], data, return_type="dataframe"
        )
        parameters = pd.Series(self.model_params)

        # Step 1, slice
        col_type = "C(hour_of_week)"
        hour_of_week_cols = [
            c
            for c in design_matrix_granular.columns
            if col_type in c and c in parameters.keys()
        ]

        # Step 2, cut out all 0s
        design_matrix_granular = design_matrix_granular[
            (design_matrix_granular[hour_of_week_cols] != 0).any(axis=1)
        ]

        cols_to_predict = list(
            set(parameters.keys()).intersection(set(design_matrix_granular.keys()))
        )
        design_matrix_granular = design_matrix_granular[cols_to_predict]
        parameters = parameters[cols_to_predict]

        # Step 3, predict
        prediction = design_matrix_granular.dot(parameters).rename(
            columns={0: "predicted_usage"}
        )
        # Step 4, put nans back in
        prediction = prediction.reindex(data.index)

        return prediction

    def json(self):
        """ Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """

        data = {
            "segment_name": self.segment_name,
            "formula": self.formula,
            "warnings": [w.json() for w in self.warnings],
            "model_params": self.model_params,
        }
        return data


class SegmentedModel(object):
    def __init__(
        self,
        segment_models,
        prediction_segment_type,
        prediction_segment_name_mapping=None,
        prediction_feature_processor=None,
        prediction_feature_processor_kwargs=None,
    ):
        self.segment_models = segment_models

        fitted_model_lookup = {
            segment_model.segment_name: segment_model
            for segment_model in segment_models
        }
        if prediction_segment_name_mapping is None:
            self.model_lookup = fitted_model_lookup
        else:
            self.model_lookup = {
                pred_name: fitted_model_lookup.get(fit_name)
                for pred_name, fit_name in prediction_segment_name_mapping.items()
            }
        self.prediction_segment_type = prediction_segment_type
        self.prediction_segment_name_mapping = prediction_segment_name_mapping
        self.prediction_feature_processor = prediction_feature_processor
        self.prediction_feature_processor_kwargs = prediction_feature_processor_kwargs

    def predict(
        self, prediction_index, temperature, **kwargs
    ):  # ignore extra args with kwargs
        prediction_segmentation = segment_time_series(
            temperature.index,
            self.prediction_segment_type,
            drop_zero_weight_segments=True,
        )

        iterator = iterate_segmented_dataset(
            temperature.to_frame("temperature_mean"),
            segmentation=prediction_segmentation,
            feature_processor=self.prediction_feature_processor,
            feature_processor_kwargs=self.prediction_feature_processor_kwargs,
            feature_processor_segment_name_mapping=self.prediction_segment_name_mapping,
        )
        predictions = {}
        for segment_name, segmented_data in iterator:
            segment_model = self.model_lookup.get(segment_name)
            if segment_model is None:
                continue
            prediction = segment_model.predict(segmented_data) * segmented_data.weight
            # NaN the zero weights and reindex
            prediction = prediction[segmented_data.weight > 0].reindex(prediction_index)
            predictions[segment_name] = prediction
        predictions = pd.DataFrame(predictions)
        result = pd.DataFrame({"predicted_usage": predictions.sum(axis=1, min_count=1)})
        return HourlyModelPrediction(result=result)

    def json(self):
        """ Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """

        def _json_or_none(obj):
            return None if obj is None else obj.json()

        data = {
            "segment_models": [_json_or_none(m) for m in self.segment_models],
            "model_lookup": {
                key: _json_or_none(val) for key, val in self.model_lookup.items()
            },
            "prediction_segment_type": self.prediction_segment_type,
            "prediction_segment_name_mapping": self.prediction_segment_name_mapping,
            "prediction_feature_processor": self.prediction_feature_processor.__name__,
        }
        return data


def filter_zero_weights_feature_processor(segment_name, segment_data):
    return segment_data[segment_data.weight > 0]


def iterate_segmented_dataset(
    data,
    segmentation=None,
    feature_processor=None,
    feature_processor_kwargs=None,
    feature_processor_segment_name_mapping=None,
):
    if feature_processor is None:
        feature_processor = filter_zero_weights_feature_processor

    if feature_processor_kwargs is None:
        feature_processor_kwargs = {}

    if feature_processor_segment_name_mapping is None:
        feature_processor_segment_name_mapping = {}

    def _apply_feature_processor(segment_name, segment_data):
        feature_processor_segment_name = feature_processor_segment_name_mapping.get(
            segment_name, segment_name
        )

        if feature_processor is not None:
            segment_data = feature_processor(
                feature_processor_segment_name, segment_data, **feature_processor_kwargs
            )
        return segment_data

    def _add_weights(data, weights):
        return pd.merge(data, weights, left_index=True, right_index=True)

    if segmentation is None:
        # spoof segment name and weights column
        segment_name = None
        weights = pd.DataFrame({"weight": 1}, index=data.index)
        segment_data = _add_weights(data, weights)

        segment_data = _apply_feature_processor(segment_name, segment_data)
        yield segment_name, segment_data
    else:
        for segment_name, segment_weights in segmentation.iteritems():
            weights = segment_weights.to_frame("weight")
            segment_data = _add_weights(data, weights)
            segment_data = _apply_feature_processor(segment_name, segment_data)
            yield segment_name, segment_data


def _get_calendar_year_coverage_warning(index):
    pass


def _get_hourly_coverage_warning(index, min_fraction_daily_coverage=0.9):
    pass


def _segment_weights_single(index):
    return pd.DataFrame({"all": 1.0}, index=index)


def _segment_weights_one_month(index):
    return pd.DataFrame(
        {
            month_name: (index.month == month_number).astype(float)
            for month_name, month_number in [
                ("jan", 1),
                ("feb", 2),
                ("mar", 3),
                ("apr", 4),
                ("may", 5),
                ("jun", 6),
                ("jul", 7),
                ("aug", 8),
                ("sep", 9),
                ("oct", 10),
                ("nov", 11),
                ("dec", 12),
            ]
        },
        index=index,
        columns=[
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],  # guarantee order
    )


def _segment_weights_three_month(index):
    return pd.DataFrame(
        {
            month_names: (index.month.map(lambda i: i in month_numbers)).astype(float)
            for month_names, month_numbers in [
                ("dec-jan-feb", (12, 1, 2)),
                ("jan-feb-mar", (1, 2, 3)),
                ("feb-mar-apr", (2, 3, 4)),
                ("mar-apr-may", (3, 4, 5)),
                ("apr-may-jun", (4, 5, 6)),
                ("may-jun-jul", (5, 6, 7)),
                ("jun-jul-aug", (6, 7, 8)),
                ("jul-aug-sep", (7, 8, 9)),
                ("aug-sep-oct", (8, 9, 10)),
                ("sep-oct-nov", (9, 10, 11)),
                ("oct-nov-dec", (10, 11, 12)),
                ("nov-dec-jan", (11, 12, 1)),
            ]
        },
        index=index,
        columns=[
            "dec-jan-feb",
            "jan-feb-mar",
            "feb-mar-apr",
            "mar-apr-may",
            "apr-may-jun",
            "may-jun-jul",
            "jun-jul-aug",
            "jul-aug-sep",
            "aug-sep-oct",
            "sep-oct-nov",
            "oct-nov-dec",
            "nov-dec-jan",
        ],  # guarantee order
    )


def _segment_weights_three_month_weighted(index):
    return pd.DataFrame(
        {
            month_names: index.month.map(
                lambda i: month_weights.get(str(i), 0.0)
            ).astype(float)
            for month_names, month_weights in [
                ("dec-jan-feb-weighted", {"12": 0.5, "1": 1, "2": 0.5}),
                ("jan-feb-mar-weighted", {"1": 0.5, "2": 1, "3": 0.5}),
                ("feb-mar-apr-weighted", {"2": 0.5, "3": 1, "4": 0.5}),
                ("mar-apr-may-weighted", {"3": 0.5, "4": 1, "5": 0.5}),
                ("apr-may-jun-weighted", {"4": 0.5, "5": 1, "6": 0.5}),
                ("may-jun-jul-weighted", {"5": 0.5, "6": 1, "7": 0.5}),
                ("jun-jul-aug-weighted", {"6": 0.5, "7": 1, "8": 0.5}),
                ("jul-aug-sep-weighted", {"7": 0.5, "8": 1, "9": 0.5}),
                ("aug-sep-oct-weighted", {"8": 0.5, "9": 1, "10": 0.5}),
                ("sep-oct-nov-weighted", {"9": 0.5, "10": 1, "11": 0.5}),
                ("oct-nov-dec-weighted", {"10": 0.5, "11": 1, "12": 0.5}),
                ("nov-dec-jan-weighted", {"11": 0.5, "12": 1, "1": 0.5}),
            ]
        },
        index=index,
        columns=[
            "dec-jan-feb-weighted",
            "jan-feb-mar-weighted",
            "feb-mar-apr-weighted",
            "mar-apr-may-weighted",
            "apr-may-jun-weighted",
            "may-jun-jul-weighted",
            "jun-jul-aug-weighted",
            "jul-aug-sep-weighted",
            "aug-sep-oct-weighted",
            "sep-oct-nov-weighted",
            "oct-nov-dec-weighted",
            "nov-dec-jan-weighted",
        ],  # guarantee order
    )


def segment_time_series(index, segment_type="single", drop_zero_weight_segments=False):
    segment_weight_func = {
        "single": _segment_weights_single,
        "one_month": _segment_weights_one_month,
        "three_month": _segment_weights_three_month,
        "three_month_weighted": _segment_weights_three_month_weighted,
    }.get(segment_type, None)

    if segment_weight_func is None:
        raise ValueError("Invalid segment type: %s" % (segment_type))

    segment_weights = segment_weight_func(index)

    if drop_zero_weight_segments:
        # keep only columns with non-zero weights
        total_weights = segment_weights.sum()
        columns_to_keep = total_weights[total_weights > 0].index.tolist()
        segment_weights = segment_weights[columns_to_keep]

    # TODO: Do something with these
    _get_hourly_coverage_warning(segment_weights)  # each model
    _get_calendar_year_coverage_warning(index)  # whole index

    return segment_weights


def fit_model_segments(segmented_dataset_dict, fit_segment):
    segment_models = [
        fit_segment(segment_name, segment_data)
        for segment_name, segment_data in segmented_dataset_dict.items()
    ]

    return segment_models
