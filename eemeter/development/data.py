import pandas as pd
from pathlib import Path
import numpy as np
from scipy.interpolate import RBFInterpolator


def clean_list(lst):
    return [x.lower().replace(" ", "_") for x in lst]


class NREL_Weather_API:  # TODO: reload data for all years
    api_key = "PBjC0msokfcSXP3SD2fqT01IShnO9ZMtbNYx4WNZ"  # get your own key from https://developer.nrel.gov/signup/  #Required
    name = "Armin+Aligholian"  # required
    email = "armin@recurve.com"  # required
    interval = "60"  # required

    attributes = "ghi,dhi,dni,wind_speed,air_temperature,cloud_type,dew_point,clearsky_dhi,clearsky_dni,clearsky_ghi"  # not required
    leap_year = "false"  # not required
    utc = "false"  # not required
    reason_for_use = "beta+testing"  # not required
    your_affiliation = "Recurve"  # not required
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

            df.columns = clean_list(df.columns)

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
    def __init__(self, grid_lag_lead_days=[-7, -1, 1, 7]):
        self.grid_lag_lead_days = grid_lag_lead_days
        self.interp = RBFInterpolator
        self.interpolated_values = {}

    def interpolate(self, df, columns=["temperature", "observed"]):
        self.df = df
        self.columns = columns
        for col in columns:
            if f"interpolated_{col}" in self.df.columns:
                self.df = self.df.drop(columns=[f"interpolated_{col}"])
            self.df[f"interpolated_{col}"] = False
        # Main method to perform the interpolation
        for col in self.columns:
            self._col_interpolation(col)

        # for those datetime that we still haven't interpolated (for the columns), we will interpolate them with pd.interpolate
        for col in self.columns:
            na_datetime = self.df.loc[self.df[col].isna()].index
            self.df[col] = self.df[col].interpolate(method="time")
            
            still_na_datetime = self.df.loc[self.df[col].isna()].index
            if len(still_na_datetime) > 0:
                self.df[col] = self.df[col].fillna(method="ffill")
                self.df[col] = self.df[col].fillna(method="bfill")

            self.df.loc[
                self.df.index.isin(na_datetime), f"interpolated_{col}"
            ] = True

        return self.df

    def _col_interpolation(self, col):
        # Method to interpolate a single column
        column_df = self.df[[col]].copy()
        # add lag and lead columns
        interp_helper_cols = []
        for shift in self.grid_lag_lead_days:
            column_df[f"{col}_day_({shift})"] = column_df[col].shift(shift * 24)
            interp_helper_cols.append(f"{col}_day_({shift})")

        column_df["hour_of_day"] = column_df.index.hour
        column_df["date"] = column_df.index.date
        # group by date
        column_df = column_df.groupby("date")
        # interpolate each day
        interpolated_datetime = pd.DataFrame([], columns=["datetime", col])
        for _, group in column_df:
            nans = group[group[col].isna()]

            # Update the helper features for previous interpolated values for previous datetimes
            group = self._update_group_values(group, col, interpolated_datetime)

            if nans.empty:
                continue

            group, interpolated_datetime = self._scenario_selector_interpolator(
                group, col, interp_helper_cols, interpolated_datetime
            )

        interpolated_datetime_local = interpolated_datetime.set_index("datetime")
        self.interpolated_values[col] = interpolated_datetime_local

        # replace nans with the interpolated values for the column
        self.df.loc[self.df.index.isin(interpolated_datetime_local.index), col] = (
            self.df[self.df.index.isin(interpolated_datetime_local.index)].index.map(
                interpolated_datetime_local[col]
            )
        )
        self.df[f"interpolated_{col}"].loc[
            self.df.index.isin(interpolated_datetime_local.index)
        ] = True

    def _scenario_selector_interpolator(
        self, group, col, interp_helper_cols, interpolated_datetime
    ):

        def joint_value_options(input_string):
            from itertools import product

            str_lst = [int(i) for i in input_string.split(" ")]
            all_possible_options = [
                list(option) for option in product([0, 1], repeat=len(str_lst))
            ]
            all_possible_options = sorted(
                [
                    option
                    for option in all_possible_options
                    if all(i <= j for i, j in zip(option, str_lst))
                ],
                reverse=True,
            )
            all_possible_options.pop()  # drop all zeros option
            return all_possible_options

        normal_datapoints = group[~group[col].isna()]
        nans = group[group[col].isna()]

        nans_dt = nans.index.values
        normal_dt = normal_datapoints.index.values

        helper_matrix = group.loc[group.index.isin(nans_dt)]
        helper_matrix_mask = 1 - helper_matrix[interp_helper_cols].isna().astype(
            int
        )  # 0,1 mask for hashing the data

        columns = helper_matrix_mask.columns.tolist()

        helper_matrix_mask["pointer"] = helper_matrix_mask.apply(
            lambda row: " ".join(row.astype(str)), axis=1
        )  # string of columns that have value or not
        helper_matrix_mask["class"] = helper_matrix_mask[columns].sum(
            axis=1
        )  # this will be used for sorting the best options
        helper_matrix_mask["datetime"] = helper_matrix.index

        datetime_lists = (
            helper_matrix_mask.groupby("pointer")["datetime"].apply(list).to_dict()
        )
        nan_groups = (
            helper_matrix_mask.groupby(["class", "pointer"])
            .size()
            .reset_index(name="counts")
        )
        nan_groups = nan_groups.sort_values(
            by=["class", "counts"], ascending=[False, False]
        )
        nan_groups[f"datetime_lists_nans"] = nan_groups["pointer"].map(datetime_lists)

        has_value = group.loc[group.index.isin(normal_dt)].reset_index()
        has_value_mask = np.array(
            1 - has_value[interp_helper_cols].isna().astype(int).values
        )

        # search through option of each nan group
        for p in nan_groups["pointer"].unique():
            row = nan_groups.loc[nan_groups["pointer"] == p]
            if not row["class"].values == 0:
                nan_pointer_options = np.array(
                    joint_value_options(row["pointer"].values[0])
                )
                filter = nan_pointer_options.sum(axis=1)
                lookup_matrix = np.dot(has_value_mask, nan_pointer_options.transpose())
                eligibles = np.where(lookup_matrix < filter, 0, 1).sum(axis=0)
                count = np.where(
                    eligibles > filter, 1, 0
                )  # check if we have enough points for RBFinterpolate
                if 1 in count:  # give us the most promissing
                    idx = list(count).index(1)
                    best_scenario = nan_pointer_options[idx]
                    # select the columns based on best scenario
                    selected_cols = [
                        c
                        for i, c in enumerate(interp_helper_cols)
                        if best_scenario[i] == 1
                    ]
                    selected_has_value_index = np.where(
                        np.where(lookup_matrix < filter, 0, 1)[:, idx] == 1
                    )
                    # get the point and value to feed interpolator
                    interp_points = has_value.loc[
                        has_value.index.isin(selected_has_value_index[0])
                    ][selected_cols]
                    # add small random noise to remove singular matrix for RBF
                    noise = np.random.standard_normal(size=interp_points.shape)
                    interp_points += noise
                    interp_values = has_value.loc[
                        has_value.index.isin(selected_has_value_index[0])
                    ][col]

                    nan_points_datetime = nan_groups.loc[nan_groups["pointer"] == p][
                        f"datetime_lists_nans"
                    ].values[0]
                    nan_points = nans.loc[nans.index.isin(nan_points_datetime)][
                        selected_cols
                    ]

                    pred = self._interpolate_points(
                        interp_points, interp_values, nan_points
                    )

                    interpolated_rows = list(zip(nan_points_datetime, pred))
                    interpolated_rows = pd.DataFrame(
                        interpolated_rows, columns=["datetime", col]
                    )
                    interpolated_datetime = pd.concat(
                        [interpolated_datetime, interpolated_rows], ignore_index=True
                    )

        return group, interpolated_datetime

    def _interpolate_points(self, points, values, nans):
        # Method to interpolate a single day
        if len(values) == 0:
            return np.nan

        interp = self.interp(points, values)
        pred = interp(nans)
        return pred

    def _update_group_values(self, group, col, interpolated_datetime):
        # Method to update the values of the group based on the previous interpolated points
        for shift in self.grid_lag_lead_days:
            if shift > 0:
                col_helper = f"{col}_day_({shift})"
                current_day_datetime = group.loc[group[col_helper].isna()].index
                #check if the shift day datetime is in interpolated_datetime
                if len(current_day_datetime) > 0:
                    #shift the datetime by the shift value backward
                    shifted_datetime = current_day_datetime - pd.Timedelta(days=shift)
                    if any(interpolated_datetime['datetime'].isin(shifted_datetime)):
                        corresponeing_current_day = interpolated_datetime.loc[
                            interpolated_datetime['datetime'].isin(shifted_datetime)
                        ]['datetime']

                        matched_current_day = corresponeing_current_day + pd.Timedelta(days=shift)
                        
                        group.loc[
                            group.index.isin(matched_current_day), col_helper
                        ] = interpolated_datetime.loc[
                            interpolated_datetime['datetime'].isin(corresponeing_current_day)
                        ][col].values
        return group

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

        self.df = df
        self.kwargs = kwargs
        if "outputs" in self.kwargs:
            self.outputs = kwargs["outputs"]
        else:
            self.outputs = ["temperature", "observed"]

        self.missing_values_amount = {}
        self.too_many_missing_data = False

        self._prepare_dataframe()

    def _prepare_dataframe(self):
        def check_datetime(df):
            # get all the columns with datetime type #TODO: check if this is the best way to do this
            datetime_columns = df.select_dtypes(include=[np.datetime64]).columns
            # check if datetime is in the columns
            if "datetime" in df.columns:
                pass
            elif "datetime" in df.index.names:
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
            if "observed" in df.columns:
                # find duplicate datetime values and remove if nan
                duplicate_dt_mask = df.duplicated(subset="datetime", keep=False)
                observed_nan_mask = df["observed"].isna()
                df = df[~(duplicate_dt_mask & observed_nan_mask)]

                # if duplicated and observed is not nan, keep the largest abs(value)
                df["abs_observed"] = df["observed"].abs()
                df = df.sort_values(
                    by=["datetime", "abs_observed"], ascending=[True, False]
                )
                df = df.drop_duplicates(subset="datetime", keep="first")
                df = df.drop(columns=["abs_observed"])

            else:
                # TODO what if there is no observed column? Could have dup datetime with different temperatures
                df = df.drop_duplicates(subset="datetime", keep="first")

            return df

        # check if datetime is in the columns
        self.df = check_datetime(self.df)

        # save the original datetime column
        self.datetime_original = self.df["datetime"]

        # fill in missing datetimes
        self.df = get_contiguous_datetime(self.df)

        # remove duplicate datetime values
        self.df = remove_duplicate_datetime(self.df)

        self.df.set_index("datetime", inplace=True)

        if "solar" in self.kwargs:
            if "metadata" in self.kwargs:
                if (
                    "station_latitude"
                    and "station_longitude" in self.kwargs["metadata"]
                ):
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
        self.df = self.df[self.outputs]

    def _interpolate(self):
        # make column of interpolated boolean if any observed or temperature is nan
        # check if in each row of the columns in output has nan values, the interpolated column will be true
        self.to_be_interpolated_columns = ["temperature", "observed"]
        if "ghi" in self.outputs:
            self.to_be_interpolated_columns += ["ghi"]

        for col in self.to_be_interpolated_columns:
            if f"interpolated_{col}" in self.df.columns:
                continue
            self.outputs += [f"interpolated_{col}"]

        #check how many nans are in the columns
        nan_numbers_cols = self.df[self.to_be_interpolated_columns].isna().sum()
        # if the number of nan is more than max_missing_hours_pct, then we we flag them

        for col in self.to_be_interpolated_columns:
            if nan_numbers_cols[col] > len(self.df) * self.max_missing_hours_pct / 100:
                if not self.too_many_missing_data:
                    self.too_many_missing_data = True
                self.missing_values_amount[col] = nan_numbers_cols[col]

        # interpolate temperature and observed values
        interp = Interpolator()
        self.df = interp.interpolate(self.df, columns=self.to_be_interpolated_columns)
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
