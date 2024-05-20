from eemeter.eemeter.models.hourly import HourlyModel
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from eemeter.eemeter.models.hourly import settings as _settings
from .data import HourlyData
import numpy as np
import pandas as pd
from typing import Optional

class HourlyOptData(HourlyData):
    def __init__(self, df: pd.DataFrame, settings : Optional[_settings.HourlySettings] = None, **kwargs: dict):  # consider solar data
        """ """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df is None:
            raise ValueError("df cannot be None")
        if not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dictionary")
        

        # Initialize settings for the features preparation
        if settings is None:
            self.settings = _settings.HourlySettings()
        else:
            self.settings = settings
        # Initialize model
        self._feature_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self.X = None
        self.y = None
        self.categorical_features = None
        self.norm_features_list = None

        self.df = df
        self.kwargs = kwargs
        if "outputs" in self.kwargs:
            self.outputs = kwargs["outputs"]
        else:
            self.outputs = ["temperature", "observed"]

        self.missing_values_amount = {}
        self.too_many_missing_data = False


        if self.df.empty:
            pass
            # raise ValueError("df cannot be empty")
        else:
            #clean the data as much as possible
            self._prepare_dataframe()
            if self.too_many_missing_data:
                pass
            else:
                self.X, self.y = self._prepare_features(self.df)
        
    def _prepare_features(self, meter_data):

        #TODO: @Armin This meter data has weather data and it should be already clean
        train_features = self.settings.TRAIN_FEATURES


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
        self.norm_features_list = [i+'_norm' for i in to_be_normalized]

        self._feature_scaler.fit(df[to_be_normalized])
        self._y_scaler.fit(df["observed"].values.reshape(-1, 1))
        # add normalized features to df
        normalized_df = pd.DataFrame(self._feature_scaler.transform(df[to_be_normalized]), index=df.index, columns=self.norm_features_list)

        df = pd.concat([df, normalized_df], axis=1)
        df["observed_norm"] = self._y_scaler.transform(df["observed"].values.reshape(-1, 1))

        return df


    def _get_feature_data(self, df): 

        new_train_features = self.norm_features_list.copy()
        if self.settings.SUPPLEMENTAL_DATA is not None:
            if 'TS_SUPPLEMENTAL' in self.settings.SUPPLEMENTAL_DATA:
                if self.settings.SUPPLEMENTAL_DATA['TS_SUPPLEMENTAL'] is not None:
                    for sup in self.settings.SUPPLEMENTAL_DATA['TS_SUPPLEMENTAL']:
                        new_train_features.append(sup)
            if 'CATEGORICAL_SUPPLEMENTAL' in self.settings.SUPPLEMENTAL_DATA:
                if 'PV_INSTALLATION_DATE' in self.settings.SUPPLEMENTAL_DATA['CATEGORICAL_SUPPLEMENTAL']:
                    self.pv_intervention_date = self.settings.SUPPLEMENTAL_DATA['CATEGORICAL_SUPPLEMENTAL']['PV_INSTALLATION_DATE']
                    self.pv_intervention_date = pd.to_datetime(self.pv_intervention_date )
                    df['has_pv'] = False
                    df.loc[df['date'] >= self.pv_intervention_date.date(), 'has_pv'] = True
                    self.categorical_features.append('has_pv')

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


class HourlyOptModel(HourlyModel):
    def __init__(
        self,
        settings : _settings.HourlySettings = None,
    ):
        # Initialize settings
        if settings is None:
            self.settings = _settings.HourlySettings()
        else:
            self.settings = settings

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

    def _fit(self, X, y):
        # Fit the model
        self._model.fit(X, y)

        self.is_fit = True

        return self

    def fit(self, X, y):

        self._fit(X,y)

        return self
    

    def _predict(self, X):
        y_predict = self._model.predict(X)
        return y_predict
    
    def predict(self, X):
        if not self.is_fit:
            raise ValueError("Model is not fit yet")
        y_predict = self._predict(X)

        return y_predict
    
    def fit_predict(self, X, y):

        self._fit(X, y)

        y_predict = self._predict(X)

        return y_predict