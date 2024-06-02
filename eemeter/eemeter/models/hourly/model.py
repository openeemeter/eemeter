#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

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

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from typing import Optional
import json

from eemeter.eemeter.models.hourly import settings as _settings
from eemeter.common.metrics import BaselineMetrics, BaselineMetricsFromDict

# from eemeter.development.data import HourlyData #TODO: import from eemeter.data


class HourlyModel:
    def __init__(
        self,
        settings : Optional[_settings.HourlySettings] = None,
    ):
        """
        """

        # Initialize settings
        if settings is None:
            self.settings = _settings.HourlySettings()
        else:
            self.settings = settings
        
        # Initialize model
        self._feature_scaler = StandardScaler()
        self._y_scaler = StandardScaler()

        self._model = ElasticNet(
            alpha = self.settings.ALPHA, 
            l1_ratio = self.settings.L1_RATIO,
            selection = self.settings.SELECTION,
            max_iter = self.settings.MAX_ITER,
            random_state = self.settings.SEED,
        )
        
        self.is_fit = False
        self.categorical_features = None
        self.norm_features_list = None
        self.baseline_metrics = None


    def fit(self, baseline_data, ignore_disqualification=False):

        # if not isinstance(baseline_data, DailyBaselineData):
        #     raise TypeError("baseline_data must be a DailyBaselineData object")
        # baseline_data.log_warnings()
        # if baseline_data.disqualification and not ignore_disqualification:
        #     raise DataSufficiencyError("Can't fit model on disqualified baseline data")
        # self.baseline_timezone = baseline_data.tz
        # self.warnings = baseline_data.warnings
        # self.disqualification = baseline_data.disqualification
        
        self._fit(baseline_data)

        # if self.error["CVRMSE"] > self.settings.cvrmse_threshold:
        #     cvrmse_warning = EEMeterWarning(
        #         qualified_name="eemeter.model_fit_metrics.cvrmse",
        #         description=(
        #             f"Fit model has CVRMSE > {self.settings.cvrmse_threshold}"
        #         ),
        #         data={"CVRMSE": self.error["CVRMSE"]},
        #     )
        #     cvrmse_warning.warn()
        #     self.disqualification.append(cvrmse_warning)

        return self


    def _fit(self, Meter_Data):
        # Initialize dataframe
        # meter_data, _ = self._prepare_dataframe(meter_data)
        meter_df = Meter_Data.df.copy()


        # Prepare feature arrays/matrices
        X, y = self._prepare_features(meter_df)

        # Fit the model
        self._model.fit(X, y)

        # Get number of model parameters
        num_parameters = (np.count_nonzero(self._model.coef_)
                               + np.count_nonzero(self._model.intercept_))
        
        # Get metrics
        meter_df = self._predict(Meter_Data)
        self.baseline_metrics = BaselineMetrics(df=meter_df, num_model_params=num_parameters)

        self.is_fit = True

        return self


    def predict(
        self,
        reporting_data,
        ignore_disqualification=False,
    ):
        """Perform initial sufficiency and typechecks before passing to private predict"""
        if not self.is_fit:
            raise RuntimeError("Model must be fit before predictions can be made.")

        # if self.disqualification and not ignore_disqualification:
        #     raise DisqualifiedModelError(
        #         "Attempting to predict using disqualified model without setting ignore_disqualification=True"
        #     )
 
        # if str(self.baseline_timezone) != str(reporting_data.tz):
        #     """would be preferable to directly compare, but
        #     * using str() helps accomodate mixed tzinfo implementations,
        #     * the likelihood of sub-hour offset inconsistencies being relevant to the daily model is low
        #     """
        #     raise ValueError(
        #         "Reporting data must use the same timezone that the model was initially fit on."
        #     )

        # if not isinstance(reporting_data, (DailyBaselineData, DailyReportingData)):
        #     raise TypeError(
        #         "reporting_data must be a DailyBaselineData or DailyReportingData object"
        #     )

        df_eval = self._predict(reporting_data)

        return df_eval


    def _predict(self, Eval_Data):
        """
        Makes model prediction on given temperature data.

        Parameters:
            df_eval (pandas.DataFrame): The evaluation dataframe.

        Returns:
            pandas.DataFrame: The evaluation dataframe with model predictions added.
        """

        df_eval = Eval_Data.df.copy()
        datetime_original = Eval_Data.datetime_original
        # # get list of columns to keep in output
        columns = df_eval.columns.tolist()
        if "datetime" in columns:
            columns.remove("datetime") # index in output, not column

        X, _ = self._prepare_features(df_eval)

        y_predict_scaled = self._model.predict(X)
        y_predict = self._y_scaler.inverse_transform(y_predict_scaled)
        y_predict = y_predict.flatten()
        df_eval["predicted"] = y_predict

        # make predicted nan if interpolated is True
        # df_eval.loc[df_eval["interpolated"], "predicted"] = np.nan #TODO: check if this is necessary

        # # remove columns not in original columns and predicted
        df_eval = df_eval[[*columns, "predicted"]]

        # reset index to original datetime index
        df_eval = df_eval.loc[datetime_original]

        return df_eval


    def _prepare_features(self, meter_data):
        """
        Initializes the meter data by performing the following operations:
        - Renames the 'model' column to 'model_old' if it exists
        - Converts the index to a DatetimeIndex if it is not already
        - Adds a 'season' column based on the month of the index using the settings.season dictionary
        - Adds a 'day_of_week' column based on the day of the week of the index
        - Removes any rows with NaN values in the 'temperature' or 'observed' columns
        - Sorts the data by the index
        - Reorders the columns to have 'season' and 'day_of_week' first, followed by the remaining columns

        Parameters:
        - meter_data: A pandas DataFrame containing the meter data

        Returns:
        - A pandas DataFrame containing the initialized meter data
        """
        #TODO: @Armin This meter data has weather data and it should be already clean
        train_features = self.settings.TRAIN_FEATURES
        lagged_features = self.settings.LAGGED_FEATURES
        window = self.settings.WINDOW

        # get categorical features
        modified_meter_data = self._add_categorical_features(meter_data)

        # normalize the data
        modified_meter_data = self._normalize_data(modified_meter_data, train_features)

        X, y = self._get_feature_data(modified_meter_data)

        return X, y


    def _add_categorical_features(self, df):
        """
        """
        # Add date, month and day of week
        # There shouldn't be any day_ or month_ columns in the dataframe
        df["date"] = df.index.date
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek

        day_cat = [
                    f"day_{i}" for i in np.arange(7) + 1
                ]
        month_cat = [
            f"month_{i}"
            for i in np.arange(12) + 1
            if f"month_{i}"
        ]
        self.categorical_features = day_cat + month_cat

        days = pd.Categorical(df["day_of_week"], categories=range(1, 8))
        day_dummies = pd.get_dummies(days, prefix="day")
        # set the same index for day_dummies as df_t
        day_dummies.index = df.index

        months = pd.Categorical(df["month"], categories=range(1, 13))
        month_dummies = pd.get_dummies(months, prefix="month")
        month_dummies.index = df.index

        df = pd.concat([df, day_dummies, month_dummies], axis=1)

        return df


    def _normalize_data(self, df, train_features):
        """
        """
        to_be_normalized = train_features.copy()
        self.norm_features_list = [i+'_norm' for i in train_features]
        #TODO: save the name of the columns and train features and categorical columns, scaler for everything
        #TODO: save all the train errors
        #TODO: save model and all the potential settings

        if not self.is_fit:
            self._feature_scaler.fit(df[to_be_normalized])
            self._y_scaler.fit(df["observed"].values.reshape(-1, 1))

        # df[self.norm_features_list] = self._feature_scaler.transform(df[to_be_normalized])
        normalized_df = pd.DataFrame(self._feature_scaler.transform(df[to_be_normalized]), index=df.index, columns=self.norm_features_list)
        df = pd.concat([df, normalized_df], axis=1)
        df["observed_norm"] = self._y_scaler.transform(df["observed"].values.reshape(-1, 1))

        return df


    def _get_feature_data(self, df): #TODO: ask Travis about window and lagged features

        new_train_features = self.norm_features_list.copy()
        if self.settings.SUPPLEMENTAL_DATA is not None:
            if 'TS_SUPPLEMENTAL' in self.settings.SUPPLEMENTAL_DATA:
                if self.settings.SUPPLEMENTAL_DATA['TS_SUPPLEMENTAL'] is not None:
                    for sup in self.settings.SUPPLEMENTAL_DATA['TS_SUPPLEMENTAL']:
                        new_train_features.append(sup)
                        
            if 'CATEGORICAL_SUPPLEMENTAL' in self.settings.SUPPLEMENTAL_DATA:
                if self.settings.SUPPLEMENTAL_DATA['CATEGORICAL_SUPPLEMENTAL'] is not None:
                    for sup in self.settings.SUPPLEMENTAL_DATA['CATEGORICAL_SUPPLEMENTAL']:
                        self.categorical_features.append(sup)      


                # for sup in self.settings.SUPPLEMENTAL_DATA['categorical_supplemental']: TODO: should we add more genral entry?
                #     self.categorical_features.append(sup)

        new_train_features.sort(reverse=True)

        # get aggregated features with agg function
        agg_dict = {f: lambda x: list(x) for f in new_train_features}

        # get the features and target for each day
        ts_feature = np.array(
            df.groupby("date").agg(agg_dict).values.tolist()
        )
        
        ts_feature = ts_feature.reshape(
            ts_feature.shape[0], ts_feature.shape[1] * ts_feature.shape[2]
        )

        # get the first categorical features for each day for each sample
        unique_dummies = df[self.categorical_features + ["date"]].groupby("date").first()

        X = np.concatenate((ts_feature, unique_dummies), axis=1)
        y = np.array(
            df.groupby("date")
            .agg({"observed_norm": lambda x: list(x)})
            .values.tolist()
        )
        y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

        return X, y


    def _get_feature_data_deprecated(self, df, window, lagged_features):
            added_features = []
            if lagged_features is not None:
                for feature in lagged_features:
                    for i in range(1, window + 1):
                        df[f"{feature}_shifted_{i}"] = df[feature].shift(i * 24)
                        added_features.append(f"{feature}_shifted_{i}")

            new_train_features = self.norm_features_list + added_features
            if self.settings.SUPPLEMENTAL_DATA is not None:
                for sup in self.settings.SUPPLEMENTAL_DATA:
                    new_train_features.append(sup)
            new_train_features.sort(reverse=True)

            # backfill the shifted features and observed to fill the NaNs in the shifted features
            df[new_train_features] = df[new_train_features].bfill()
            df["observed_norm"] = df["observed_norm"].bfill()

            # get aggregated features with agg function
            agg_dict = {f: lambda x: list(x) for f in new_train_features}

            # get the features and target for each day
            ts_feature = np.array(
                df.groupby("date").agg(agg_dict).values.tolist()
            )
            
            ts_feature = ts_feature.reshape(
                ts_feature.shape[0], ts_feature.shape[1] * ts_feature.shape[2]
            )

            # get the first categorical features for each day for each sample
            unique_dummies = df[self.categorical_features + ["date"]].groupby("date").first()

            X = np.concatenate((ts_feature, unique_dummies), axis=1)
            y = np.array(
                df.groupby("date")
                .agg({"observed_norm": lambda x: list(x)})
                .values.tolist()
            )
            y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

            return X, y


    def to_dict(self):
        feature_scaler = {}
        for i, key in enumerate(self.settings.TRAIN_FEATURES):
            feature_scaler[key] = [self._feature_scaler.mean_[i], self._feature_scaler.scale_[i]]

        y_scaler = [self._y_scaler.mean_, self._y_scaler.scale_]
        
        params = _settings.SerializeModel(
            SETTINGS = self.settings,
            COEFFICIENTS = self._model.coef_.tolist(),
            INTERCEPT = self._model.intercept_.tolist(),
            FEATURE_SCALER = feature_scaler,
            CATAGORICAL_SCALER = None,
            Y_SCALER= y_scaler,
            BASELINE_METRICS = self.baseline_metrics,
        )

        return params.model_dump()


    def to_json(self):
        return json.dumps(self.to_dict())


    @classmethod
    def from_dict(cls, data):
        # get settings
        settings = _settings.HourlySettings(**data.get("SETTINGS"))

        # initialize model class
        model_cls = cls(settings=settings)

        features = list(data.get("FEATURE_SCALER").keys())

        # set feature scaler
        feature_scaler_values= list(data.get("FEATURE_SCALER").values())
        feature_scaler_mean = [i[0] for i in feature_scaler_values]
        feature_scaler_scale = [i[1] for i in feature_scaler_values]

        model_cls._feature_scaler.mean_ = np.array(feature_scaler_mean)
        model_cls._feature_scaler.scale_ = np.array(feature_scaler_scale)
        
        # set y scaler
        y_scaler_values = data.get("Y_SCALER")

        model_cls._y_scaler.mean_ = np.array(y_scaler_values[0])
        model_cls._y_scaler.scale_ = np.array(y_scaler_values[1])

        # set model
        model_cls._model.coef_ = np.array(data.get("COEFFICIENTS"))
        model_cls._model.intercept_ = np.array(data.get("INTERCEPT"))

        model_cls.is_fit = True

        # set baseline metrics
        model_cls.baseline_metrics = BaselineMetricsFromDict(data.get("BASELINE_METRICS"))

        # info = data.get("info")
        # model_cls.params = DailyModelParameters(
        #     submodels=data.get("submodels"),
        #     info=info,
        #     settings=settings,
        # )

        # def deserialize_warnings(warnings):
        #     if not warnings:
        #         return []
        #     warn_list = []
        #     for warning in warnings:
        #         warn_list.append(
        #             EEMeterWarning(
        #                 qualified_name=warning.get("qualified_name"),
        #                 description=warning.get("description"),
        #                 data=warning.get("data"),
        #             )
        #         )
        #     return warn_list

        # model_cls.disqualification = deserialize_warnings(
        #     info.get("disqualification")
        # )
        # model_cls.warnings = deserialize_warnings(info.get("warnings"))
        # model_cls.baseline_timezone = info.get("baseline_timezone")

        return model_cls


    @classmethod
    def from_json(cls, str_data):
        return cls.from_dict(json.loads(str_data))


    def plot(
        self,
        df_eval,
        ax=None,
        title=None,
        figsize=None,
        temp_range=None,
    ):
        """Plot a model fit.

        Parameters
        ----------
        ax : :any:`matplotlib.axes.Axes`, optional
            Existing axes to plot on.
        title : :any:`str`, optional
            Chart title.
        figsize : :any:`tuple`, optional
            (width, height) of chart.
        with_candidates : :any:`bool`
            If True, also plot candidate models.
        temp_range : :any:`tuple`, optionl
            Temperature range to plot

        Returns
        -------
        ax : :any:`matplotlib.axes.Axes`
            Matplotlib axes.
        """
        try:
            from eemeter.eemeter.models.daily.plot import plot
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting.")

        # TODO: pass more kwargs to plotting function

        plot(self, self._predict(df_eval.df))