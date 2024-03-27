import pandas as pd
import numpy as np

# import torch
import os
from pathlib import Path
import time as time
import multiprocessing as mp
import pickle

from applied_data_science.bigquery.data import Meter_Data
from eemeter import eemeter as em
from eemeter.common.metrics import BaselineMetrics as Metrics
from sklearn.preprocessing import StandardScaler

def clean_list(lst):
    return [x.lower().replace(" ", "_") for x in lst]


class NREL_Weather_API:
    api_key = "PBjC0msokfcSXP3SD2fqT01IShnO9ZMtbNYx4WNZ"  # get your own key from https://developer.nrel.gov/signup/  #Required
    name = "Armin+Aligholian"  # required
    email = "armin@recurve.com"  # required
    interval = "60"  # required

    attributes = "ghi,dhi,dni,wind_speed,air_temperature,cloud_type,dew_point,clearsky_dhi,clearsky_dni,clearsky_ghi"  # not required
    leap_year = "false"  # not required
    utc = "true"  # not required, but don't change
    reason_for_use = "beta+testing"  # not required
    your_affiliation = "Recurve"  # not required
    mailing_list = "false"  # not required

    cache = Path("/app/.recurve_cache/data/MCE/MCE_weather_stations")
    use_cache = True

    round_minutes_method = "floor"  # [None, floor, ceil, round]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.cache.mkdir(parents=True, exist_ok=True)

    def get_data(self, lat, lon, years=[2017, 2020]):
        data_path = self.cache / f"{lat}_{lon}.pkl"
        if data_path.exists() and self.use_cache:
            df = pd.read_pickle(data_path)

        else:
            years = list(range(min(years), max(years) + 1))

            df = self.query_API(lat, lon, years)

            if self.use_cache:
                df.to_pickle(data_path)

        df.columns = clean_list(df.columns)

        if self.round_minutes_method == "floor":
            df["datetime"] = df["datetime"].dt.floor("H")
        elif self.round_minutes_method == "ceil":
            df["datetime"] = df["datetime"].dt.ceil("H")
        elif self.round_minutes_method == "round":
            df["datetime"] = df["datetime"].dt.round("H")

        df = df.set_index("datetime")

        return df

    def query_API(self, lat, lon, years):
        leap_year = self.leap_year
        interval = self.interval
        utc = self.utc
        api_key = self.api_key
        name = self.name
        email = self.email

        year_df = []
        for year in years:
            year = str(year)

            url = self._generate_url(
                lat, lon, year, leap_year, interval, utc, api_key, name, email
            )
            df = pd.read_csv(url, skiprows=2)

            # Set the time index in the pandas dataframe:
            # set datetime using the year, month, day, and hour
            df["datetime"] = pd.to_datetime(
                df[["Year", "Month", "Day", "Hour", "Minute"]]
            )

            df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute"])
            df = df.dropna()

            year_df.append(df)

        # merge the dataframes for different years
        df = pd.concat(year_df, axis=0)

        return df

    def _generate_url(
        self, lat, lon, year, leap_year, interval, utc, api_key, name, email
    ):
        query = f"?wkt=POINT({lon}%20{lat})&names={year}&interval={interval}&api_key={api_key}&full_name={name}&email={email}&utc={utc}"

        if year == "2021":
            # details: https://developer.nrel.gov/docs/solar/nsrdb/psm3-2-2-download/
            url = f"https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.csv{query}"

        elif year in [str(i) for i in range(1998, 2021)]:
            # details: https://developer.nrel.gov/docs/solar/nsrdb/psm3-download/
            url = f"https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv{query}"

        else:
            print("Year must be between 1998 and 2021")
            url = None

        return url


class MCE_Data_Loader_Test:
    def __init__(self, settings=None):
        if settings is not None:
            self.settings = settings
            self.window = settings.WINDOW
            self.train_features = settings.TRAIN_FEATURES
            self.lagged_features = settings.LAGGED_FEATURES
        else:
            self.window = None
        self.months = [
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
        ]
        self.non_neccessary_columns = ['jan_train', 'jan_test',
       'feb_train', 'feb_test', 'mar_train', 'mar_test', 'apr_train',
       'apr_test', 'may_train', 'may_test', 'jun_train', 'jun_test',
       'jul_train', 'jul_test', 'aug_train', 'aug_test', 'sep_train',
       'sep_test', 'oct_train', 'oct_test', 'nov_train', 'nov_test',
       'dec_train', 'dec_test','fill_flag','global_horizontal_uv_irradiance_(280-400nm)',
       'global_horizontal_uv_irradiance_(295-385nm)']

    def _get_solar_data(self, metadata, years):
        lat = metadata["lat"]
        lon = metadata["lon"]
        # print(f'{metadata['sid']}')

        nrel_weather = NREL_Weather_API(use_cache=True)
        df = nrel_weather.get_data(lat, lon, years)
        # change the temperature column name
        df = df.rename(columns={"temperature": "temp_NRSDB"})

        # convert to kWh
        for feature in [
            "ghi",
            "dni",
            "dhi",
            "clearsky_dhi",
            "clearsky_dni",
            "clearsky_ghi",
        ]:
            if feature in df.columns:
                df[feature] /= 1000

        return df

    def _get_df(self, metadata, df, T_type="NOAA"):
        # get unique years from sdf from 'start' column
        years = np.unique(df["start"].dt.year)

        # get solar data
        sdf = self._get_solar_data(metadata, years)

        # assign temperature column
        if T_type == "NOAA":
            pass
        elif T_type == "NRSDB":
            df["temperature"] = df["temp_NRSDB"]
        else:
            raise ValueError("T_type must be either NOAA or NRSDB")

        sdf = sdf.drop(columns=["temp_NRSDB"])

        # merge site data with solar data
        df = df.merge(sdf, left_on="start", right_index=True, how="left")
        df = df.set_index("start")
        df = df.sort_index()


        exclude_cols = []
        for month in self.months:
            exclude_cols.append(f"{month}_train")
            exclude_cols.append(f"{month}_test")

        # Replace NaNs
        # TODO: can this be avoided?
        valid_cols = ~df.columns.isin(exclude_cols)

        # Replace NaNs with the previous value
        # df.loc[:, valid_cols] = df.loc[:, valid_cols].fillna(method='ffill')

        # Replace Nans with interpolated values
        # TODO: we can interpolate better (current day and days bordering NaNs)
        df.loc[:, valid_cols] = df.loc[:, valid_cols].interpolate(
            method="linear", limit_direction="both"
        )

        return df

    def _get_df_old(self, metadata, df, T_type="NOAA"):
        df = self._get_df(metadata, df, T_type)

        # TODO: is any of this necessary?
        # align the dataset to local time starts from 0 and ends with 24
        start_idx = np.where(df.index.hour == 0)[0][0]
        end_idx = np.where(df.index.hour == 23)[0][-1]
        df = df.iloc[start_idx : end_idx + 1]
        df = df.groupby("date").filter(
            lambda x: len(x) == 24
        )  # TODO: check if this is necessary

        # drop list of columns
        # drop_cols = []
        # for month in months:
        #     drop_cols.append(f'{month}_train')
        #     drop_cols.append(f'{month}_test')

        # df = df.drop(columns=drop_cols)

        return df

    def get_all_cleaned_data(self, metadata, df):

        sid = metadata["sid"]

        # Get weather station data
        df = self._get_df_old(metadata, df, T_type="NOAA")

        df_train = []
        df_test = []
        for month in self.months:

            # select the days that have data for the month
            idx_train = np.argwhere(np.isfinite(df[f"{month}_train"].values)).flatten()

            idx_finite = np.argwhere(
                np.isfinite(df.iloc[idx_train]["observed"].values)
            ).flatten()
            if len(idx_finite) == 0:
                continue

            idx_train = idx_train[idx_finite]

            idx_test = np.argwhere(np.isfinite(df[f"{month}_test"].values)).flatten()

            idx_finite = np.argwhere(
                np.isfinite(df.iloc[idx_test]["observed"].values)
            ).flatten()
            if len(idx_finite) == 0:
                continue

            idx_test = idx_test[idx_finite]

            df_train_temp = df.iloc[idx_train]
            df_train_temp = df_train_temp.groupby("date").filter(
                lambda x: len(x) == 24
            )  # This will drop the last day if it is not 24 hours
            df_test_temp = df.iloc[idx_test]
            df_test_temp = df_test_temp.groupby("date").filter(
                lambda x: len(x) == 24
            )  # This will drop the last day if it is not 24 hours

            # group by date and get all the values of train_features as a 2d array both for train and test

            for train_test in ["train", "test"]:
                if train_test == "train":
                    df_t = df_train_temp.copy()
                else:
                    df_t = df_test_temp.copy()

                # set CalTRACK 3.0 hourly output as oeem column
                df_t["caltrack"] = df_t[f"{month}_{train_test}"]

                # de-select none neccessary columns
                df_t = df_t.drop(columns=self.non_neccessary_columns)

                # time delta UTC to local
                td = df_t.index.min() - df_t["start_local"].iloc[0]

                # create new datetimes based on min max of the index
                new_datetimes = pd.date_range(
                    start=df_t.index.min(),
                    end=df_t.index.max(),
                    freq="H",
                )

                # Reindex and backfill to add those missing days in the dataset
                df_t = df_t.reindex(new_datetimes)
                df_t["start_local"] = df_t.index - td
                # backfill the missing values of all columns except the date and categorical features
                df_t = df_t.fillna(
                    method="bfill"
                )
            
                if train_test == "train":
                    df_train.append(df_t)
                else:
                    df_test.append(df_t)

        return sid, df_train, df_test

    def get_all_cleaned_data_new(self, metadata, df):

        sid = metadata["sid"]

        # Get weather station data
        df = self._get_df_old(metadata, df, T_type="NOAA")

        train_datasets = {"X": [], "y": [], "y_scalar": []}
        test_datasets = {"X": [], "y": []}
        df_train = []
        df_test = []
        for month in self.months:

            # select the days that have data for the month
            idx_train = np.argwhere(np.isfinite(df[f"{month}_train"].values)).flatten()

            idx_finite = np.argwhere(
                np.isfinite(df.iloc[idx_train]["observed"].values)
            ).flatten()
            if len(idx_finite) == 0:
                continue

            idx_train = idx_train[idx_finite]

            idx_test = np.argwhere(np.isfinite(df[f"{month}_test"].values)).flatten()

            idx_finite = np.argwhere(
                np.isfinite(df.iloc[idx_test]["observed"].values)
            ).flatten()
            if len(idx_finite) == 0:
                continue

            idx_test = idx_test[idx_finite]

            df_train_temp = df.iloc[idx_train]
            df_train_temp = df_train_temp.groupby("date").filter(
                lambda x: len(x) == 24
            )  # This will drop the last day if it is not 24 hours
            df_test_temp = df.iloc[idx_test]
            df_test_temp = df_test_temp.groupby("date").filter(
                lambda x: len(x) == 24
            )  # This will drop the last day if it is not 24 hours

            # group by date and get all the values of train_features as a 2d array both for train and test
            scaler = StandardScaler()
            y_scaler = StandardScaler()
            for train_test in ["train", "test"]:
                if train_test == "train":
                    df_t = df_train_temp.copy()
                else:
                    df_t = df_test_temp.copy()

                # set CalTRACK 3.0 hourly output as oeem column
                df_t["caltrack"] = df_t[f"{month}_{train_test}"]

                # de-select none neccessary columns
                df_t = df_t.drop(columns=self.non_neccessary_columns)

                # time delta UTC to local
                td = df_t.index.min() - df_t["start_local"].iloc[0]

                # create new datetimes based on min max of the index
                new_datetimes = pd.date_range(
                    start=df_t.index.min(),
                    end=df_t.index.max(),
                    freq="H",
                )

                # Reindex and backfill to add those missing days in the dataset
                df_t = df_t.reindex(new_datetimes)
                df_t["start_local"] = df_t.index - td
                # backfill the missing values of all columns except the date and categorical features
                df_t = df_t.bfill()

                df_t["date"] = df_t.index.date
                df_t["month"] = df_t.index.month
                df_t["day_of_week"] = df_t.index.dayofweek

                day_cat = [
                            f"day_{i}" for i in np.arange(7) + 1
                        ]
                month_cat = [
                    f"month_{i}"
                    for i in np.arange(12) + 1
                    if f"month_{i}"
                ]
                self.categorical_features = day_cat + month_cat

                days = pd.Categorical(df_t["day_of_week"], categories=range(1, 8))
                day_dummies = pd.get_dummies(days, prefix="day")
                # set the same index for day_dummies as df_t
                day_dummies.index = df_t.index

                months = pd.Categorical(df_t["month"], categories=range(1, 13))
                month_dummies = pd.get_dummies(months, prefix="month")
                month_dummies.index = df_t.index

                df_t = pd.concat([df_t, day_dummies, month_dummies], axis=1)

                to_be_normalized = self.train_features.copy()
                self.norm_features_list = [i+'_norm' for i in self.train_features]
                #TODO: save the name of the columns and train features and categorical columns, scaler for everything
                #TODO: save all the train errors
                #TODO: save model and all the potential settings

                if train_test == "train":
                    scaler.fit(df_t[to_be_normalized])
                    y_scaler.fit(df_t["observed"].values.reshape(-1, 1))

                df_t[self.norm_features_list] = scaler.transform(df_t[to_be_normalized])
                df_t["observed_norm"] = y_scaler.transform(df_t["observed"].values.reshape(-1, 1))

                added_features = []
                if self.lagged_features is not None:
                    for feature in self.lagged_features:
                        for i in range(1, self.window + 1):
                            df_t[f"{feature}_shifted_{i}"] = df_t[feature].shift(i * 24)
                            added_features.append(f"{feature}_shifted_{i}")

                new_train_features = self.norm_features_list + added_features

                new_train_features.sort(reverse=True)

                # backfill the shifted features and observed to fill the NaNs in the shifted features
                df_t[new_train_features] = df_t[new_train_features].bfill()
                df_t["observed_norm"] = df_t["observed_norm"].bfill()

                # get aggregated features with agg function
                agg_dict = {f: lambda x: list(x) for f in new_train_features}

                # get the features and target for each day
                ts_feature = np.array(
                    df_t.groupby("date").agg(agg_dict).values.tolist()
                )
                ts_feature = ts_feature.reshape(
                    ts_feature.shape[0], ts_feature.shape[1] * ts_feature.shape[2]
                )

                # get the first categorical features for each day for each sample
                unique_dummies = df_t[self.categorical_features + ["date"]].groupby("date").first()

                X = np.concatenate((ts_feature, unique_dummies), axis=1)
                y = np.array(
                    df_t.groupby("date")
                    .agg({"observed_norm": lambda x: list(x)})
                    .values.tolist()
                )
                y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])

                if train_test == "train":
                    df_train.append(df_t)
                    train_datasets["X"].append(X)
                    train_datasets["y"].append(y)
                    train_datasets["y_scalar"].append(y_scaler)
                else:
                    df_test.append(df_t)
                    test_datasets["X"].append(X)
                    test_datasets["y"].append(y)
                    
        return sid, train_datasets, test_datasets, df_train, df_test


def save_features(arglist):
    sid, solar_meter, subsample_num, sd, metadata, settings = arglist
    main_path = f"/app/.recurve_cache/mce_3_yr_precovid/MCE_features"
    main_path = f"{main_path}/{solar_meter}/{subsample_num}"
    # check if the path exists
    if not os.path.exists(main_path):
        os.makedirs(main_path)
        print(f"Created path: {main_path}")

    data_loader = MCE_Data_Loader_Test(settings)
    sid, train_datasets, test_datasets, df_trains, df_tests = data_loader.get_all_cleaned_data_new(metadata, sd)
    for k in range(len(df_trains)):
        X_train, y_train, y_scaler = train_datasets["X"][k], train_datasets["y"][k], train_datasets["y_scalar"][k]
        train_observed = df_trains[k]["observed"].values
        X_test, y_test = test_datasets["X"][k], test_datasets["y"][k]
        test_observed = df_tests[k]["observed"].values
        iqr = df_trains[k]["observed"].quantile(0.75) - df_trains[k]["observed"].quantile(0.25)
        data_details = {
            "sid": sid,
            "solar_meter": solar_meter,
            "subsample_num": subsample_num,
            "k-fold": k
        }
        # save all the above to a pickle file
        save_dict = {
            "X_train": X_train,
            "y_train": y_train,
            "y_scaler": y_scaler,
            "train_observed": train_observed,
            "X_test": X_test,
            "y_test": y_test,
            "test_observed": test_observed,
            "iqr": iqr,
            'data_details': data_details
        }
        path = f"{main_path}/{sid}_{k}.pkl"
        # check if the file exists
        if not os.path.exists(path):        
            with open(path, "wb") as f:
                pickle.dump(save_dict, f)

def get_features(data_arglist):
    sid, kfold, solar_meter, subsample_num = data_arglist
    main_path = f"/app/.recurve_cache/mce_3_yr_precovid/MCE_features"
    main_path = f"{main_path}/{solar_meter}/{subsample_num}"
    path = f"{main_path}/{sid}_{kfold}.pkl"
    status = True
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except:
        status = False
        data = None    
    return status, path, sid, kfold, solar_meter, subsample_num, data

def train_dec_from_loaded(arglist):
    status, path, sid, kfold, solar_meter, subsample_num, data, settings = arglist
    #load the data
    tic = time.time()
    if not status:
        print(f"Data for {path} not found")
        return sid, None, None, None, None
    else:
        load_data_time = time.time() - tic
        
        X_train, y_train, y_scaler = data["X_train"], data["y_train"], data["y_scaler"]
        train_observed = data["train_observed"]
        X_test, y_test = data["X_test"], data["y_test"]
        test_observed = data["test_observed"]
        iqr = data["iqr"]

        errors = {'train': [], 'test': []}

        calc_tic = time.time()
        model = em.HourlyModelTest(settings=settings)
        model.fit(X_train, y_train)
        fit_time = time.time() - calc_tic
        
        train_y_pred = model.predict(X_train, y_scaler)
        test_y_pred = model.predict(X_test, y_scaler)
        
        # df_train = pd.DataFrame()
        # df_train['observed'] = train_observed
        # df_train['predicted'] = train_y_pred

        # df_test = pd.DataFrame()
        # df_test['observed'] = test_observed
        # df_test['predicted'] = test_y_pred

        # train_metric = Metrics(df=df_train, num_model_params=model.num_model_params)
        # test_metric = Metrics(df=df_test, num_model_params=model.num_model_params)      

        train_rmse = np.sqrt(np.mean((train_observed - train_y_pred)**2))
        train_pnrmse = train_rmse / iqr
        errors['train'].append(train_pnrmse)

        test_rmse = np.sqrt(np.mean((test_observed - test_y_pred)**2))
        test_pnrmse = test_rmse / iqr
        errors['test'].append(test_pnrmse)

        total_time = time.time() - tic

        id_details = {
            "sid": sid,
            "solar_meter": solar_meter,
            "subsample_num": subsample_num,
            "kfold": kfold
        }
        time_details = {
            "load_data_time": load_data_time,
            "fit_time": fit_time,
            "total_time": total_time
        }
    return id_details, time_details, errors

class Population_Run_Features:
    def __init__(self, **kwargs):
        self.dataset = 'mce_3_yr_precovid'
        self.cache_dir = Path("/app/.recurve_cache/").resolve()
        self.main_folder = "MCE_features"
        self.files_path = []
        self.results = None
        self.arglist = None
        subsamples, solar_meters, settings = kwargs["subsamples"], kwargs["solar_meters"], kwargs["settings"]
        self.subsamples = subsamples
        self.solar_meters = solar_meters
        self.n_kfold = 12
        self.settings = settings
    
    def _get_files_list(self):
        self.files_path = []
        # get the unique ids
        for subsample_num in self.subsamples:
            for solar_meter in self.solar_meters:
                #make the path considering boolian and integer values for subsample_num and solar_meter
                path = self.cache_dir / self.dataset / self.main_folder / str(solar_meter) / str(subsample_num)
                if not path.exists():
                    raise ValueError(f"Path {path} does not exist")
                else:
                    #files path
                    files = list(path.glob("*.pkl"))
                    self.files_path.extend(files)                 

    def _create_arglist(self):
        self.arglist = []
        for file in self.files_path:
            sid = file.stem.rsplit('_', 1)[0]
            kfold = file.stem.split('_')[-1]
            kfold = int(kfold)
            solar_meter, subsample_num = file.parts[-3:-1]
            self.arglist.append((sid, kfold, solar_meter, subsample_num))

    def _load_data(self):
        self._get_files_list()
        self._create_arglist()
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            self.arglist = pool.map(get_features, self.arglist)
        
        self.arglist = [(*d, self.settings) for d in self.arglist]

    def run(self):
        self._load_data()
        self.results = []
        # add setting to all the all_data_res
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            self.results = pool.map(train_dec_from_loaded, self.arglist)



def train_dec(arglist):
    sid, sd, metadata, data_loader, settings = arglist

    tic = time.time()
    data_loader = MCE_Data_Loader_Test()
    sid, df_trains, df_tests = data_loader.get_all_cleaned_data(metadata, sd)
    load_data_tac = time.time()
    errors = {'train': {
        'new_model': [],
        'oeem': []
    }, 'test': {
        'new_model': [],
        'oeem': []
    }} 
    calc_tic = time.time()
    avg_fit_time =0
    for k in range(len(df_trains)):
        tic2 = time.time()
        model = em.HourlyModel(settings=settings)
        model.fit(df_trains[k])
        tak2 = time.time()
        avg_fit_time += tak2 - tic2
        num_model_parameters = model.baseline_metrics.num_model_params
        iqr = model.baseline_metrics.observed.iqr

        for train_test in ['train', 'test']:
            if train_test == 'train':
                new_df = df_trains[k].copy()
            else:
                new_df = df_tests[k].copy()
          
            new_df = model.predict(new_df)
            
            metrics = Metrics(df=new_df, num_model_params=num_model_parameters)
            obs = metrics.observed
            ct = metrics.caltrack
                        
            err = metrics.rmse / iqr
            errors[train_test]['new_model'].append(err)

            ct_pnrmse = np.sqrt(np.mean((obs.values - ct.values)**2))
            ct_pnrmse = ct_pnrmse / iqr
            errors[train_test]['oeem'].append(ct_pnrmse)

    avg_fit_time = avg_fit_time / len(df_trains)
    load_data_time = load_data_tac - tic
    tak = time.time()
    total_time = tak - tic
    calc_time = tak - calc_tic
    return sid, total_time, calc_time, errors, avg_fit_time, load_data_time


def train_dec_new(arglist):
    sid, sd, metadata, data_loader, settings = arglist
    tic = time.time()
    data_loader = MCE_Data_Loader_Test(settings)
    sid, train_datasets, test_datasets, df_trains, df_tests = data_loader.get_all_cleaned_data_new(metadata, sd)
    load_data_tac = time.time()
    errors = {'train': {
        'new_model': [],
        'oeem': []
    }, 'test': {
        'new_model': [],
        'oeem': []
    }} 
    calc_tic = time.time()
    avg_fit_time =0
    for k in range(len(df_trains)):
        tic2 = time.time()
        model = em.HourlyModelTest(settings=settings)
        X_train, y_train, y_scaler = train_datasets["X"][k], train_datasets["y"][k], train_datasets["y_scalar"][k]
        X_test, y_test = test_datasets["X"][k], test_datasets["y"][k]
        model.fit(X_train, y_train)
        
        tak2 = time.time()
        avg_fit_time += tak2 - tic2
        iqr = df_trains[k]["observed"].quantile(0.75) - df_trains[k]["observed"].quantile(0.25)

        for train_test in ['train', 'test']:
            if train_test == 'train':
                y_pred = model.predict(X_train, y_scaler)
                df_trains[k]["predicted"] = y_pred
                new_df = df_trains[k].copy()
            else:
                y_pred = model.predict(X_test, y_scaler)
                df_tests[k]["predicted"] = y_pred
                new_df = df_tests[k].copy()
          
            metrics = Metrics(df=new_df, num_model_params=model.num_model_params)
            obs = metrics.observed
            ct = metrics.caltrack
                        
            err = metrics.rmse / iqr
            errors[train_test]['new_model'].append(err)

            ct_pnrmse = np.sqrt(np.mean((obs.values - ct.values)**2))
            ct_pnrmse = ct_pnrmse / iqr
            errors[train_test]['oeem'].append(ct_pnrmse)

    avg_fit_time = avg_fit_time / len(df_trains)
    load_data_time = load_data_tac - tic
    tak = time.time()
    total_time = tak - tic
    calc_time = tak - calc_tic
    return sid, total_time, calc_time, errors, avg_fit_time, load_data_time

class Population_Run:
    def __init__(self, settings, data):
        self.results = None
        self.settings = settings
        self.data = data
    
    def _error_calc(self):
        tr_nm_err = []
        te_nm_err = []
        tr_ct_err = []
        te_ct_err = []
        for res in self.results:
            tr_nm_err.append(np.mean(res[3]['train']['new_model']))
            te_nm_err.append(np.mean(res[3]['test']['new_model']))

            tr_ct_err.append(np.mean(res[3]['train']['oeem']))
            te_ct_err.append(np.mean(res[3]['test']['oeem']))
        
        return {
            'train': {
                'new_model': tr_nm_err,
                'oeem': tr_ct_err
            },
            'test': {
                'new_model': te_nm_err,
                'oeem': te_ct_err
            }
        }

    def run(self):
        # get data
        meta = self.data.df["meta"]
        subsample_df = self.data.df["meter"]
        ids = subsample_df.index.unique()

        arglist = []
        for sid in ids:
            lat = meta.loc[meta.index == sid].iloc[0]["station_latitude"]
            lon = meta.loc[meta.index == sid].iloc[0]["station_longitude"]
            sd = subsample_df.loc[sid].copy()
            metadata = {"lat": lat, "lon": lon, "sid": sid}
            data_loader = MCE_Data_Loader_Test()
            arglist.append([sid, sd, metadata, data_loader, self.settings])

        
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            self.results = pool.map(train_dec_new, arglist) #TODO: change this to train_dec

        self.errors = self._error_calc()

    def save_errors(self, path):
        # save the error to pkl
        with open(path, "wb") as f:
            pickle.dump(self.errors, f)

    def save_results(self, path):
        # save the results to pkl
        with open(path, "wb") as f:
            pickle.dump(self.results, f)
        self.results = None

    def save_config(self, path):
        # save the config to pkl
        with open(path, "wb") as f:
            pickle.dump(self.config, f)
        self.config = None
