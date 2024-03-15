
import pandas as pd
import numpy as np

# import torch
import os
from pathlib import Path

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
    def __init__(self, config):
        self.config = config
        self.window = config["window"] # nubmer of lag days 
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

                # time delta UTC to local
                td = df_t.index.min() - df_t["start_local"].iloc[0]

                # create new datetimes based on min max of the index
                new_datetimes = pd.date_range(
                    start=df_t.index.min() - pd.DateOffset(days=self.window),
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



def model_training_dec_local(arglst):
    tic = time.time()
    data_loader, sd, metadata, config, sid = arglst
    train_datasets, test_datasets, df_trains, df_tests = [[], [], [], []]

    segment = config['segment']

    try:
        if config["window"] is not None:
            # get test train data
            sid, train_datasets, test_datasets, df_trains, df_tests = (
                data_loader.get_window_test_train(metadata, sd)
            )
        else:
            # get test train data
            sid, train_datasets, test_datasets, df_trains, df_tests = (
                data_loader.get_window_to_one(metadata, sd)
            )

        model_detail = config["model_detail"]
        status = True
        model = model_detail["model"]
        model_settings = model_detail.copy()
        del model_settings["model"]
        # let's train the model for each month
        months = [
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
        calc_tic = time.time()
        for k in range(len(months)):
            month = months[k]

            if segment == "monthly":

                X_train, y_train, y_scalar = train_datasets["X"][k], train_datasets["y"][k], train_datasets["y_scalar"][k]
                y_train_pred = np.zeros(y_train.shape)

                X_test, y_test = test_datasets["X"][k], test_datasets["y"][k]
                y_test_pred = np.zeros(y_test.shape)
                w_train = train_datasets["w"][k]

                train_segments = pd.DataFrame(train_datasets['X'][k][:][:,-12:])
                train_grouped= train_segments.groupby(train_segments.columns.tolist())

                test_segments = pd.DataFrame(test_datasets['X'][k][:][:,-12:])
                test_grouped= test_segments.groupby(test_segments.columns.tolist())
                for name, group in train_grouped:
                    lr = model(**model_settings)

                    # start = 2*config['hours_shifted']
                    X_train_segment = X_train[group.index]
                    X_test_segment = X_test[test_grouped.get_group(name).index]
                    if config['window'] is not None:
                        X_train_segment = X_train_segment[:,0:-12]
                        X_test_segment = X_test_segment[:,0:-12]
                    else:
                        pass
                        # start, end = -12-24, -24    
                        # X_train_segment = np.concatenate((X_train_segment[:,:start], X_train_segment[:,end:]), axis=1)
                        # X_test_segment = np.concatenate((X_test_segment[:,:start], X_test_segment[:,end:]), axis=1)

                    y_train_segment = y_train[group.index]

                    if w_train is None:
                        lr.fit(X_train_segment, y_train_segment)
                    else:
                        lr.fit(X_train_segment, y_train_segment, sample_weight=w_train[group.index])
                    
                    y_test_segment = y_test[test_grouped.get_group(name).index]

                    y_train_pred[group.index] = lr.predict(X_train_segment)

                    y_test_pred[test_grouped.get_group(name).index] = lr.predict(X_test_segment)
            else:
                lr = model(**model_settings)
                if config['window'] is None:
                    X_train, y_train = train_datasets["X"][k], train_datasets["y"][k]
                else:
                    X_train, y_train, y_scalar = train_datasets["X"][k], train_datasets["y"][k], train_datasets["y_scalar"][k]
                X_test, y_test = test_datasets["X"][k], test_datasets["y"][k]
                w_train = train_datasets["w"][k]
                
                if w_train is None:
                    lr.fit(X_train, y_train)
                else:
                    lr.fit(X_train, y_train, sample_weight=w_train)

                y_train_pred = lr.predict(X_train)
                y_test_pred = lr.predict(X_test)
                if config['window'] is not None:
                    y_train_pred = y_scalar.inverse_transform(y_train_pred)
                    y_test_pred = y_scalar.inverse_transform(y_test_pred)

            y_train_pred = y_train_pred.flatten()
            y_test_pred = y_test_pred.flatten()

            df_trains[k]["new_model"] = y_train_pred
            df_tests[k]["new_model"] = y_test_pred

            # make a new column for the oeem model
            train_model_out_name = f"{month}_train"
            test_model_out_name = f"{month}_test"
            df_trains[k]["oeem"] = df_trains[k][train_model_out_name]
            df_tests[k]["oeem"] = df_tests[k][test_model_out_name]

            df_trains[k] = df_trains[k][config["output"]]
            df_tests[k] = df_tests[k][config["output"]]
        tak = time.time()
        total_time = tak - tic
        calc_time = tak - calc_tic
    except:
        status = False
        sid = sid
        df_trains = pd.DataFrame([])
        df_tests = pd.DataFrame([])
        total_time = 0
        calc_time = 0

    return status, sid, total_time, calc_time, df_trains, df_tests


def model_training_dec_local_OEEM(arglst):
    tic = time.time()
    data_loader, sd, metadata, config, sid = arglst
    train_datasets, test_datasets, df_trains, df_tests = [[], [], [], []]
    status = True

    try:

        sid, train_datasets, test_datasets, df_trains, df_tests = (
            data_loader.get_window_test_train(metadata, sd)
        )

        # let's train the model for each month
        months = [
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
        calc_tic = time.time()
        for k in range(len(months)):
            month = months[k]

            dftr = df_trains[k].copy()
            dfts = df_tests[k].copy()
            dftr.index = dftr.index.tz_localize('UTC').tz_convert('US/Pacific')
            dfts.index = dfts.index.tz_localize('UTC').tz_convert('US/Pacific')

            baseline_data = HourlyBaselineData.from_series(dftr['observed'], dftr['temperature'], is_electricity_data=True)
            reporting_data = HourlyReportingData.from_series(dfts['observed'], dfts['temperature'], is_electricity_data=True)
            
            model = HourlyModel()
            
            y_train_pred = model.fit(baseline_data)
            y_train_pred = model.predict(baseline_data)['predicted_usage']
            y_test_pred = model.predict(reporting_data)['predicted_usage']

            y_train_pred = y_train_pred
            y_test_pred = y_test_pred

            df_trains[k]["new_model"] = y_train_pred.values
            df_tests[k]["new_model"] = y_test_pred.values

            train_model_out_name = f"{month}_train"
            test_model_out_name = f"{month}_test"
            df_trains[k]["oeem"] = df_trains[k][train_model_out_name]
            df_tests[k]["oeem"] = df_tests[k][test_model_out_name]

            df_trains[k] = df_trains[k][config["output"]]
            df_tests[k] = df_tests[k][config["output"]]

    except:
        status = False
        sid = sid
        df_trains = pd.DataFrame([])
        df_tests = pd.DataFrame([])
    tak = time.time()
    total_time = tak - tic
    calc_time = tak - calc_tic
    return status, sid, total_time, calc_time, df_trains, df_tests


class Error_Calc:
    def __init__(self):
        self.train_error = {"pnrmse": {}, "cvrmse": {}, "rmse": {}}
        self.test_error = {"pnrmse": {}, "cvrmse": {}, "rmse": {}}

    def set_iqr(self, observed):
        self.iqr = observed.quantile(0.75) - observed.quantile(0.25)

    def rmse_calc(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

    def pnrmse_calc(self, y_true, y_pred, iqr):
        return self.rmse_calc(y_true, y_pred) / iqr

    def cvrmse_calc(self, y_true, y_pred):
        return self.rmse_calc(y_true, y_pred) / np.mean(y_true)

    def calculate_total_error(self, df_trains, df_tests):
        for i in range(len(df_trains)):
            df_train = df_trains[i]
            df_test = df_tests[i]
            iq3 = df_train["observed"].quantile(0.75)
            iq1 = df_train["observed"].quantile(0.25)
            iqr = iq3 - iq1

            for model in ["oeem", "new_model"]:
                if model not in self.test_error["pnrmse"]:
                    self.test_error["pnrmse"][model] = []
                    self.test_error["cvrmse"][model] = []
                    self.test_error["rmse"][model] = []

                    self.train_error["pnrmse"][model] = []
                    self.train_error["cvrmse"][model] = []
                    self.train_error["rmse"][model] = []

                self.test_error["pnrmse"][model].append(
                    self.pnrmse_calc(df_test["observed"], df_test[model], iqr)
                )
                self.test_error["cvrmse"][model].append(
                    self.cvrmse_calc(df_test["observed"], df_test[model])
                )
                self.test_error["rmse"][model].append(
                    self.rmse_calc(df_test["observed"], df_test[model])
                )

                self.train_error["pnrmse"][model].append(
                    self.pnrmse_calc(df_train["observed"], df_train[model], iqr)
                )
                self.train_error["cvrmse"][model].append(
                    self.cvrmse_calc(df_train["observed"], df_train[model])
                )
                self.train_error["rmse"][model].append(
                    self.rmse_calc(df_train["observed"], df_train[model])
                )

        # change to dataframe
        for key in self.test_error:
            self.test_error[key] = pd.DataFrame(self.test_error[key])
            self.train_error[key] = pd.DataFrame(self.train_error[key])


def cal_error_decor(res):
    error_calc = Error_Calc()
    status, id, total_time, calc_time, df_trains, df_tests = res
    error_calc.calculate_total_error(df_trains, df_tests)
    te_er = {}
    tr_er = {}
    for key in error_calc.test_error:
        te_er[key + "_mean"] = error_calc.test_error[key].mean(axis=0)
        te_er[key + "_std"] = error_calc.test_error[key].std(axis=0)
        tr_er[key + "_mean"] = error_calc.train_error[key].mean(axis=0)
        tr_er[key + "_std"] = error_calc.train_error[key].std(axis=0)
    te_er = pd.DataFrame(te_er).transpose()
    tr_er = pd.DataFrame(tr_er).transpose()
    return status, id, total_time, calc_time, tr_er, te_er


class Population_Error_Calc:
    def __init__(self, results):
        self.results = results
        self.error_calc()
        self.get_error()

    def error_calc(self):
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            self.error_calcs = pool.map(cal_error_decor, self.results)

    def get_error(self):
        oeem = {"train": pd.DataFrame(), "test": pd.DataFrame()}
        new_model = {"train": pd.DataFrame(), "test": pd.DataFrame()}
        total_time_sum = 0
        calc_time_sum = 0
        for i in range(len(self.error_calcs)):
            for model in ["oeem", "new_model"]:
                status, id, total_time, calc_time, tr_er, te_er = self.error_calcs[i]
                if status:
                    temp_tr = pd.DataFrame(
                        [tr_er[model].values], columns=tr_er[model].index
                    )
                    temp_te = pd.DataFrame(
                        [te_er[model].values], columns=te_er[model].index
                    )
                    if model == "oeem":
                        oeem["train"] = pd.concat(
                            [oeem["train"], temp_tr], ignore_index=True
                        )
                        oeem["test"] = pd.concat(
                            [oeem["test"], temp_te], ignore_index=True
                        )
                    elif model == "new_model":
                        new_model["train"] = pd.concat(
                            [new_model["train"], temp_tr], ignore_index=True
                        )
                        new_model["test"] = pd.concat(
                            [new_model["test"], temp_te], ignore_index=True
                        )
                        total_time_sum += total_time
                        calc_time_sum += calc_time
        self.errors = {"oeem": oeem, "new_model": new_model}
        self.total_time = total_time_sum
        self.calc_time = calc_time_sum


class Population_Run:
    def __init__(self, model_mode="new_model"):
        self.results = None
        self.model_mode = model_mode

    def set_data(self, data):
        self.data = data

    def run(self, config):
        self.config = config
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
            data_loader = MCE_Data_Loader(self.config)
            arglist.append([data_loader, sd, metadata, self.config, sid])

        # run the model
        if self.model_mode == "oeem":
            with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                self.results = pool.map(model_training_dec_local_OEEM, arglist)
                print("oeem")
        elif self.model_mode == "new_model":
            with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                self.results = pool.map(model_training_dec_local, arglist)
        else:
            print("model mode not recognized")

        PEC = Population_Error_Calc(self.results)
        self.errors = PEC.errors
        self.total_time = PEC.total_time
        self.calc_time = PEC.calc_time

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
