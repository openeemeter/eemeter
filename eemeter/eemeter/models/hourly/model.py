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
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



class HourlyModel:
    def __init__(
        self,
        settings=None,
    ):
        """
        """

        # Initialize settings
        if settings is None:
            self.settings = self._default_settings()
        else:
            self.settings = settings
        
        # Initialize model
        self._model_initiation()
        
        self.error = {
            "wRMSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "CVRMSE": np.nan,
            "PNRMSE": np.nan,
        }

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

    def _fit(self, meter_data):
        # Initialize dataframe
        self.train_status = "fitting"
        X, y = self._prepare_features(meter_data)

        # Begin fitting
        self.regressor.fit(X, y)
        
        # self._get_error_metrics()      

        # self.params = self._create_params_from_fit_model()
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

        return self._predict(reporting_data)

    def _predict(self, df_eval):
        """
        Makes model prediction on given temperature data.

        Parameters:
            df_eval (pandas.DataFrame): The evaluation dataframe.

        Returns:
            pandas.DataFrame: The evaluation dataframe with model predictions added.
        """
        self.train_status = "predicting"
        X, _ = self._prepare_features(df_eval)

        y_predict_scaled = self.regressor.predict(X)
        y_predict = self.y_scaler.inverse_transform(y_predict_scaled)
        y_predict = y_predict.flatten()
        df_eval["new_model"] = y_predict

        return df_eval

    def to_dict(self):
        return self.params.model_dump()

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        settings = data.get("settings")
        daily_model = cls(settings=settings)
        info = data.get("info")
        daily_model.params = DailyModelParameters(
            submodels=data.get("submodels"),
            info=info,
            settings=settings,
        )

        def deserialize_warnings(warnings):
            if not warnings:
                return []
            warn_list = []
            for warning in warnings:
                warn_list.append(
                    EEMeterWarning(
                        qualified_name=warning.get("qualified_name"),
                        description=warning.get("description"),
                        data=warning.get("data"),
                    )
                )
            return warn_list

        daily_model.disqualification = deserialize_warnings(
            info.get("disqualification")
        )
        daily_model.warnings = deserialize_warnings(info.get("warnings"))
        daily_model.baseline_timezone = info.get("baseline_timezone")
        daily_model.is_fit = True
        return daily_model

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

    def _create_params_from_fit_model(self):
        submodels = {}
        for key, submodel in self.model.items():
            temperature_constraints = {
                "T_min": submodel.T_min,
                "T_max": submodel.T_max,
                "T_min_seg": submodel.T_min_seg,
                "T_max_seg": submodel.T_max_seg,
            }
            submodels[key] = DailySubmodelParameters(
                coefficients=submodel.named_coeffs,
                temperature_constraints=temperature_constraints,
                f_unc=submodel.f_unc,
            )
        params = DailyModelParameters(
            submodels=submodels,
            settings=self.settings.to_dict(),
            info={
                "error": self.error,
                "baseline_timezone": str(self.baseline_timezone),
                "disqualification": [dq.json() for dq in self.disqualification],
                "warnings": [warning.json() for warning in self.warnings],
            },
        )

        return params

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
        train_features = self.settings["train_features"].copy()
        window = self.settings["window"]
        # set temp as the main lag feature
        if "lagged_features" in self.settings:
            lagged_features = self.settings["lagged_features"]
        else:
            lagged_features = ["temperature"]

        #get categorical features
        modified_meter_data = self._add_categorical_features(meter_data)
        # normalize the data
        modified_meter_data = self._normalize_data(modified_meter_data, train_features)

        X, y = self._get_feature_data(modified_meter_data, train_features, window, lagged_features)

        return X, y

    def _add_categorical_features(self, df):
        """
        """
        # Add season and day of week
        # There shouldn't be any day_ or month_ columns in the dataframe
        df["date"] = df.index.date
        df["day_of_week"] = df.index.dayofweek

        day_cat = [
                    f"day_{i}" for i in np.arange(7) + 1
                ]
        month_cat = [
            f"month_{i}"
            for i in np.arange(12) + 1
            if f"month_{i}"
        ]
        self.cat_features = day_cat + month_cat

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

        if self.train_status == "fitting":
            scaler = StandardScaler()
            y_scalar = StandardScaler()
            scaler.fit(df[to_be_normalized])
            y_scalar.fit(df["observed"].values.reshape(-1, 1))

            self.scaler = scaler
            self.y_scaler = y_scalar

        df[self.norm_features_list] = self.scaler.transform(df[to_be_normalized])
        df["observed_norm"] = self.y_scaler.transform(df["observed"].values.reshape(-1, 1))
        return df

    def _get_feature_data(self, df, train_features, window, lagged_features):
        added_features = []
        for feature in train_features:
            if feature in lagged_features:
                for i in range(1, window):
                    df[f"{feature}_shifted_{i}"] = df[feature].shift(i * 24)
                    added_features.append(f"{feature}_shifted_{i}")

        new_train_features = self.norm_features_list + added_features
        if self.settings['supplimental_data']:
            new_train_features.append('supplimental_data')
        new_train_features.sort(reverse=True)

        # backfill the shifted features and observed to fill the NaNs in the shifted features
        df[train_features] = df[new_train_features].fillna(method="bfill")
        df["observed_norm"] = df["observed_norm"].fillna(method="bfill")

        # exclude the first window days, we need 365 days
        #TODO: I think this is not necessary
        df = df.iloc[(window - 1) * 24 :]
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
        unique_dummies = df[self.cat_features + ["date"]].groupby("date").first()

        X = np.concatenate((ts_feature, unique_dummies), axis=1)
        y = np.array(
            df.groupby("date")
            .agg({"observed_norm": lambda x: list(x)})
            .values.tolist()
        )
        y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

        return X, y

    def _get_error_metrics(self): #TODO: set the error metrics
        """
        
        """

        N = 0


        resid = np.hstack(resid)
        obs = np.hstack(obs)

        self.error["wRMSE"] = np.sqrt(wSSE / N)
        self.error["RMSE"] = RMSE = np.mean(resid**2) ** 0.5
        self.error["MAE"] = np.mean(np.abs(resid))
        self.error["CVRMSE"] = RMSE / np.mean(obs)
        self.error["PNRMSE"] = RMSE / np.diff(np.quantile(obs, [0.25, 0.75]))[0]
    
    def _model_initiation(self):
        """
        Initializes the model by setting the settings and creating the model object.

        Parameters:
        - settings: A dictionary containing the settings for the model

        Returns:
        - None
        """

        self.model = ElasticNet
        self.model_kwarg = self.settings["model_kwarg"]
        self.regressor = self.model(**self.model_kwarg)

        self.error = {
            "wRMSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "CVRMSE": np.nan,
            "PNRMSE": np.nan,
        }
        self.is_fit = False
        self.train_status = "unfitted"
        self.cat_features = None
        self.norm_features_list = None
        self.scaler = None
        self.y_scaler = None

    def _default_settings(self):

        train_features = ['ghi']
        if 'temperature' not in train_features:
            train_features.append('temperature')
        operational_time = False
        # analytic_features = ['GHI', 'Temperature', 'DHI', 'DNI', 'Relative Humidity', 'Wind Speed', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type']
        lagged_features = ['temperature'] # 'ghi'

        output = ['start_local', 'temperature', 'ghi', 'clearsky_ghi', 'observed', 'new_model', 'month']
        model_kwarg = {'alpha': 0.1, 'l1_ratio': 0.1, 'random_state': 1}
        window = 1
       
        settings = {'train_features': train_features, 'model_kwarg': model_kwarg, 'window': window,
                'lagged_features': lagged_features, 'output': output,  'supplimental_data': operational_time}
        
        return settings
