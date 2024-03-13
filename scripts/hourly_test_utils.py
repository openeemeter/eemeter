
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