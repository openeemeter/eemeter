from pathlib import Path

import numpy as np
import pandas as pd
import copy

from scipy.interpolate import RBFInterpolator


class NREL_Weather_API:  # TODO: reload data for all years
    api_key = "---"  # get your own key from https://developer.nrel.gov/signup/  #Required
    name = "---"  # required
    email = "---"  # required
    interval = "60"  # required

    attributes = "ghi,dhi,dni,wind_speed,air_temperature,cloud_type,dew_point,clearsky_dhi,clearsky_dni,clearsky_ghi"  # not required
    leap_year = "false"  # not required
    utc = "false"  # not required
    reason_for_use = "---"  # not required
    your_affiliation = "---"  # not required
    mailing_list = "false"  # not required

    # cache = Path("/app/.recurve_cache/data/MCE/MCE_weather_stations")
    cache = Path("/app/.recurve_cache/data/MCE/Weather_stations")

    use_cache = True

    round_minutes_method = "floor"  # [None, floor, ceil, round]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.cache.mkdir(parents=True, exist_ok=True)

    def get_data(self, lat, lon, years=[2017, 2021]):
        data_path = self.cache / f"{lat}_{lon}.pkl"
        if data_path.exists() and self.use_cache:
            df = pd.read_pickle(data_path)

        else:
            years = list(range(min(years), max(years) + 1))

            df = self.query_API(lat, lon, years)

            df.columns = [x.lower().replace(" ", "_") for x in df.columns]

            if self.round_minutes_method == "floor":
                df["datetime"] = df["datetime"].dt.floor("H")
            elif self.round_minutes_method == "ceil":
                df["datetime"] = df["datetime"].dt.ceil("H")
            elif self.round_minutes_method == "round":
                df["datetime"] = df["datetime"].dt.round("H")

            df = df.set_index("datetime")

            if self.use_cache:
                df.to_pickle(data_path)

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


class Interpolator:
    def __init__(self, **kwargs):
        super().__init__()
        if "n_cor_idx" in kwargs:
            self.n_cor_idx = kwargs["n_cor_idx"]
        else:
            self.n_cor_idx = 6
        self.lags = 24 * 7 * 2 + 1  # TODO: make this a parameter

    def interpolate(self, df, columns=["temperature", "observed"]):
        self.df = df
        self.columns = columns
        for col in columns:
            if f"interpolated_{col}" in self.df.columns:
                self.df = self.df.drop(columns=[f"interpolated_{col}"])
            self.df[f"interpolated_{col}"] = False
        # Main method to perform the interpolation
        for col in self.columns: #TODO: bad meters should be removed by now
            if col == 'observed':
                missing_frac = self.df[col].isna().sum() / len(self.df)
                self.n_cor_idx = int(np.max([6, np.round((4.012*np.log(missing_frac) + 24.38)/2, 0)*2]))
            else:
                self.n_cor_idx = 6
            self._col_interpolation(col)

        # for those datetime that we still haven't interpolated (for the columns), we will interpolate them with pd.interpolate
        for col in self.columns:
            na_datetime = self.df.loc[self.df[col].isna()].index
            if len(na_datetime) > 0:
                # interpolate the missing values
                self.df[col] = self.df[col].interpolate(method="time")
            # check if we still have missing values
            still_na_datetime = self.df.loc[self.df[col].isna()].index
            if len(still_na_datetime) > 0:
                self.df[col] = self.df[col].fillna(method="ffill")
                self.df[col] = self.df[col].fillna(method="bfill")

            #TODO: we can check if we have similar values multiple times back to back, if yes, raise a warning
            self.df.loc[self.df.index.isin(na_datetime), f"interpolated_{col}"] = True

        return self.df

    def _col_interpolation(self, col):
        helper_df = self.df.copy()
        # Calculate the correlation of col with its lags and leads
        results = {
            i: helper_df[col].autocorr(lag=i) for i in range(-self.lags, self.lags)
        }
        results = pd.DataFrame(
            results.values(), index=results.keys(), columns=["autocorr"]
        )
        # remove zero
        results = results[results.index != 0]
        results = results.sort_values(by="autocorr", ascending=False).head(
            self.n_cor_idx
        )

        # interpolate and update the values
        check = True
        while check:
            helper_columns = []
            for shift in results.index:
                if shift < 0:
                    shift_type = "lag"
                else:
                    shift_type = "lead"

                self.df[f"{col}_{shift_type}_{shift}"] = self.df[f"{col}"].shift(-shift)
                helper_columns.append(f"{col}_{shift_type}_{shift}")

            nan_idx_before_interp = self.df.index[self.df[f"{col}"].isna()]
            # fill the missing values with the mean of the selected lag lead
            self.df.loc[nan_idx_before_interp, f"{col}"] = self.df.loc[
                nan_idx_before_interp, helper_columns
            ].mean(axis=1)
            # check if we still have missing values
            nan_idx_after_interp = self.df.index[self.df[f"{col}"].isna()]

            interpolated_datetime_local = nan_idx_before_interp.difference(
                nan_idx_after_interp
            )
            # print("interpolated with model: ", interpolated_datetime_local.shape)

            self.df[f"interpolated_{col}"].loc[
                self.df.index.isin(interpolated_datetime_local)
            ] = True

            if interpolated_datetime_local.shape[0] == 0:
                check = False

        nan_idx = self.df.index[self.df[f"{col}"].isna()]
        # check if we still have missing values
        if self.df[f"{col}"].isna().sum() > 0:  # TODO: make this more robust
            self.df[f"{col}"] = self.df[f"{col}"].interpolate(
                method="time", limit_direction="both"
            )

        self.df.loc[nan_idx, f"interpolated_{col}"] = True
        self.df.drop(columns=helper_columns, inplace=True)


class HourlyData:
    max_missing_hours_pct = 10

    def __init__(self, df: pd.DataFrame, **kwargs: dict):  # consider solar data
        """ """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if df is None:
            raise ValueError("df cannot be None")
        if not isinstance(kwargs, dict):
            raise TypeError("kwargs must be a dictionary")

        self.to_be_interpolated_columns = []
        self.interp = None
        self.outputs = []

        self.df = df
        self.kwargs = copy.deepcopy(kwargs)
        if "outputs" in self.kwargs:
            self.outputs = copy.deepcopy(self.kwargs["outputs"])
        else:
            self.outputs = ["temperature", "observed"]

        self.missing_values_amount = {}
        self.too_many_missing_data = False

        if self.df.empty:
            raise ValueError("df cannot be empty")
        self._prepare_dataframe()

    def _prepare_dataframe(self):
        def check_datetime(df):
            # get all the columns with datetime type #TODO: check if this is the best way to do this
            datetime_columns = df.select_dtypes(include=[np.datetime64]).columns
            # check if datetime is in the columns
            if "datetime" in df.columns:
                pass
            elif "datetime" in df.index.names or "start_local" in df.index.names:
                df["datetime"] = df.index
                df = df.reset_index(drop=True)
            elif "start_local" in df.columns:
                df["datetime"] = df["start_local"]
                df = df.drop(columns=["start_local"])
            elif len(datetime_columns) > 0:
                df["datetime"] = df[datetime_columns[0]]
                df = df.drop(columns=[datetime_columns[0]])
            else:
                raise ValueError("datetime column not found")

            # reset index to ensure datetime is not the index
            df = df.reset_index()
            return df

        def get_contiguous_datetime(df):
            # get earliest datetime and latest datetime
            # make earliest start at 0 and latest end at 23, this ensures full days
            earliest_datetime = (
                df["datetime"].min().replace(hour=0, minute=0, second=0, microsecond=0)
            )
            latest_datetime = (
                df["datetime"].max().replace(hour=23, minute=0, second=0, microsecond=0)
            )

            # create a new index with all the hours between the earliest and latest datetime
            complete_dt = pd.date_range(
                start=earliest_datetime, end=latest_datetime, freq="H"
            ).to_frame(index=False, name="datetime")

            # merge meter data with complete_dt
            df = complete_dt.merge(df, on="datetime", how="left")
            df["date"] = df["datetime"].dt.date
            df["hour_of_day"] = df["datetime"].dt.hour

            return df

        def remove_duplicate_datetime(df):

            duplicate_dt_mask = df[df.index.duplicated()].index.copy()
            # find index to keep
            index_to_keep = df.index.difference(duplicate_dt_mask)
            temp_df = df.loc[duplicate_dt_mask, :]
            # groupby index and select the first value with the actual value
            temp_df = temp_df.groupby(temp_df.index).apply(
                lambda x: x["observed"].sort_values().iloc[0]
            )
            # merge with the original data
            df = pd.concat([df.loc[index_to_keep, :], temp_df], axis=0).sort_index()

            return df

        # check if datetime is in the columns
        self.df = check_datetime(self.df)

        # save the original datetime column
        self.datetime_original = self.df["datetime"]

        # fill in missing datetimes
        self.df = get_contiguous_datetime(self.df)

        self.df.set_index("datetime", inplace=True)

        # remove duplicate datetime values
        self.df = remove_duplicate_datetime(self.df)

        # self.df.set_index("datetime", inplace=True)

        if "metadata" in self.kwargs:
            if ("solar" in self.kwargs) & (not self.df.columns.isin(["ghi"]).any()):
                # add solar data
                lat_exists = "station_latitude" in self.kwargs["metadata"]
                lon_exists = "station_longitude" in self.kwargs["metadata"]
                if lat_exists and lon_exists:
                    self.station_latitude = self.kwargs["metadata"]["station_latitude"]
                    self.station_longitude = self.kwargs["metadata"][
                        "station_longitude"
                    ]

                else:  # TODO: add eeweather to get the station_latitude and station_longitude
                    # if just have lat and lon of the meter
                    raise ValueError(
                        "station_latitude and station_longitude are not in metadata"
                    )

                self._add_solar_data()

        self.df = self._interpolate()
        # add pv start date here to avoid interpolating the pv start date
        if "metadata" in self.kwargs:
            if "pv_start" in self.kwargs["metadata"]:
                self.pv_start = self.kwargs["metadata"]["pv_start"]
                if self.pv_start is not None:
                    self.pv_start = pd.to_datetime(self.pv_start).date()

                self._add_pv_start_date(self.pv_start)

        self.df = self.df[self.outputs]
        # remove any duplicated columns
        self.df = self.df.loc[
            :, ~self.df.columns.duplicated()
        ]  # TODO: check why we even get these duplicates

    def _interpolate(self):
        # make column of interpolated boolean if any observed or temperature is nan
        # check if in each row of the columns in output has nan values, the interpolated column will be true
        if "to_be_interpolated_columns" in self.kwargs:
            self.to_be_interpolated_columns = self.kwargs[
                "to_be_interpolated_columns"
            ].copy()
            self.outputs += [f"{col}" for col in self.to_be_interpolated_columns if col not in self.outputs]
        else:
            self.to_be_interpolated_columns = ["temperature", "observed"]

        # for col in self.outputs:
        #     if col not in self.to_be_interpolated_columns: #TODO: this might be diffrent for supplemental data
        #         self.to_be_interpolated_columns += [col]

        # #TODO: remove this in the actual implementation, this is just for CalTRACK testing
        # if 'model' in self.outputs:
        #     self.to_be_interpolated_columns += ['model']

        for col in self.to_be_interpolated_columns:
            if f"interpolated_{col}" in self.df.columns:
                continue
            self.outputs += [f"interpolated_{col}"]

        # check how many nans are in the columns
        nan_numbers_cols = self.df[self.to_be_interpolated_columns].isna().sum()
        # if the number of nan is more than max_missing_hours_pct, then we we flag them
        #TODO: this should be as a part of disqualification and warning/error logs
        for col in self.to_be_interpolated_columns:
            if nan_numbers_cols[col] > len(self.df) * self.max_missing_hours_pct / 100:
                if not self.too_many_missing_data:
                    self.too_many_missing_data = True
                self.missing_values_amount[col] = nan_numbers_cols[col]

        # we can add kwargs to the interpolation class like: inter_kwargs = {"n_cor_idx": self.kwargs["n_cor_idx"]}
        self.interp = Interpolator()

        self.df = self.interp.interpolate(
            df=self.df, columns=self.to_be_interpolated_columns
        )
        return self.df

    def _get_location_solar_data(self):
        # get unique years from sdf from 'start' column
        # years = np.unique(self.df.index.year)
        years = [2017, 2021]  # TODO: get all years saved them then call as needed

        lat = self.station_latitude
        lon = self.station_longitude

        nrel_weather = NREL_Weather_API(use_cache=True)
        solar_df = nrel_weather.get_data(lat, lon, years)
        # change the temperature column name
        solar_df = solar_df.rename(columns={"temperature": "temp_NRSDB"})

        # convert to kWh
        for feature in [
            "ghi",
            "dni",
            "dhi",
            "clearsky_dhi",
            "clearsky_dni",
            "clearsky_ghi",
        ]:
            if feature in solar_df.columns:
                solar_df[feature] /= 1000

        return solar_df

    def _add_solar_data(
        self, T_type="NOAA"
    ):  # TODO: should we consider NRSDB temp at all?
        # get solar data
        sdf = self._get_location_solar_data()

        # assign temperature column
        if T_type == "NOAA":
            pass
        elif T_type == "NRSDB":
            self.df["temperature"] = self.df["temp_NRSDB"]
        else:
            raise ValueError("T_type must be either NOAA or NRSDB")

        self.sdf = sdf.drop(columns=["temp_NRSDB"])

        # merge site data with solar data on index
        self.df = self.df.merge(sdf, left_index=True, right_index=True, how="left")

    def _add_pv_start_date(self, pv_start, model_type="TS"):
        if pv_start is None:
            pv_start = self.df.index.date.min()

        if "ts" in model_type.lower() or "time" in model_type.lower():
            self.df["has_pv"] = 0
            self.df.loc[self.df["date"] >= pv_start, "has_pv"] = 1

        else:
            self.df["has_pv"] = False
            self.df.loc[self.df["date"] >= pv_start, "has_pv"] = True
