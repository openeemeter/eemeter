from gridmeter._utils.data_processing_settings import Data_Settings
from gridmeter._utils import const as _const
import pandas as pd
import numpy as np
import time


class Data:
    def __init__(self, settings: Data_Settings | None = None):
        if settings is None:
            settings = Data_Settings()

        self.settings = settings

        self.loadshape = None
        self.features = None

        # TODO: let's make id the index
        self.excluded_ids = pd.DataFrame(columns=["id", "reason"])


    def extend(self, other):
        """
            Extend the current Data instance with the Data instance(s) in other by concatenating the features and loadshape dataframes.
        """
        if not isinstance(other, list):
            other = [other]

        for data_instance in other:
            if isinstance(data_instance, Data):
                if self.features is not None and data_instance.features is not None:
                    self.features = pd.concat([self.features, data_instance.features])
                if self.loadshape is not None and data_instance.loadshape is not None:
                    self.loadshape = pd.concat([self.loadshape, data_instance.loadshape])
            else:
                raise TypeError("All elements in other must be instances of Data")
            

    def _find_groupby_columns(self) -> list:
        """
        Create the list of columns to be grouped by based on the time_period selected in Settings.

        Time_period : hour => group by (id, hour)
        Time_period : month => group by (id, month)
        Time_period : hourly_day_of_week => group by (id, day_of_week, hour)
        Time_period : weekday_weekend => group by (id, weekday_weekend)
        Time_period : season_day_of_week => group by (id, season, day_of_week)
        Time_period : season_hourly_weekday_weekend => group by (id, season, weekday_weekend, hour)

        """
        cols = ["id"]

        for period in _const.unique_time_periods:
            if period in self.settings.TIME_PERIOD:
                cols.append(period)

        return cols

    def _add_index_columns_from_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add hour column
        if "hour" in self.settings.TIME_PERIOD:
            df["hour"] = df['datetime'].dt.hour

        # Add month column
        if "month" in self.settings.TIME_PERIOD:
            df["month"] = df['datetime'].dt.month

        # Add day_of_week column
        if "day_of_week" in self.settings.TIME_PERIOD:
            df["day_of_week"] = df['datetime'].dt.dayofweek

        # Add day_of_year column
        if "day_of_year" in self.settings.TIME_PERIOD:
            df["day_of_year"] = df['datetime'].dt.dayofyear

        # Add weekday_weekend column
        if "weekday_weekend" in self.settings.TIME_PERIOD:
            df["weekday_weekend"] = df['datetime'].dt.dayofweek

            # Setting the ordering to weekday, weekend
            df["weekday_weekend"] = (
                df["weekday_weekend"]
                .map(self.settings.WEEKDAY_WEEKEND._NUM_DICT)
                .map(self.settings.WEEKDAY_WEEKEND._ORDER)
            )

        # Add season column
        if "season" in self.settings.TIME_PERIOD:
            df["season"] = df['datetime'].dt.month.map(self.settings.SEASON._NUM_DICT).map(
                self.settings.SEASON._ORDER
            )

        return df

    def _create_values_for_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing values in the dataframe based on the settings.
        
        - create a new dataframe with id's and correct time column
        - join on new df and old
        - interpolate nan values


        """

        if self.settings.INTERPOLATE_MISSING:
            unique_ids = df['id'].unique()
            unique_time_counts = None

            if self.settings.TIME_PERIOD is None: # loadshape type dataframe
                unique_time_counts = df["time"].max()

            else: # timeseries type dataframe
                unique_time_counts = _const.time_period_row_counts[self.settings.TIME_PERIOD]
                    

            time_values = range(1, unique_time_counts + 1)
            # Create the expected dataframe having the correct number of timestamps for each id    
            df_expected = pd.DataFrame({
                'id': np.repeat(unique_ids, unique_time_counts),
                'time': np.tile(time_values, len(unique_ids))
            })

            # Join the expected dataframe with the input dataframe
            df = df_expected.merge(df, how='left', on=['id', 'time'])

        return df


    def _validate_format_loadshape(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check columns missing in loadshape_df
        expected_columns = ["id", "time", "loadshape"]
        missing_columns = [c for c in expected_columns if c not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns in loadshape_df: {missing_columns}")

        # Check if all values are present in the columns as required
        # Else update the values via interpolation if missing, also ignore duplicates if present

        # loadshape df has the "time" column, whereas timeseries df has the "datetime" column
        subset_columns = expected_columns[:-1]

        # To eliminate duplicates, sort the values by loadshape and the keep the first (i.e. the lowest) value
        df = df.sort_values(by='loadshape', key=abs).drop_duplicates(subset=subset_columns, keep="first")

        # Check that the minimum time counts per id is consistent for the input loadshape_df
        unique_time_counts = df["time"].max()
        unique_time_counts_per_id = df.groupby("id")["time"].nunique()

        if self.settings.INTERPOLATE_MISSING:
            if self.settings.TIME_PERIOD is None:
                # for loadshape type dataframe
                # if I input a loadshape, I don't want to have to tell it the time_period I used
                # The time column should directly be pivoted, and the error checking should ensure that the number of values is consistent per meter

                invalid_ids = unique_time_counts_per_id[
                    unique_time_counts_per_id
                    < unique_time_counts * self.settings.MIN_DATA_PCT_REQUIRED
                ].index.tolist()
                excluded_ids = pd.DataFrame(
                    {
                        "id": invalid_ids,
                        "reason": "Unique time counts per id don't have the minimum time counts required",
                    }
                )
                self.excluded_ids = pd.concat(
                    [self.excluded_ids, excluded_ids], ignore_index=True
                )

            else:
                # Check that the number of missing values is less than the threshold
                for id, group in df.groupby("id"):
                    if (
                        group.count().min()
                        < self.settings.MIN_DATA_PCT_REQUIRED
                        * _const.time_period_row_counts[self.settings.TIME_PERIOD]
                    ):
                        # throw out meters with missing values and record them, do not throw error

                        excluded_ids = pd.DataFrame(
                            {
                                "id": [id],
                                "reason": [
                                    "missing minimum number of values in loadshape_df"
                                ],
                            }
                        )

                        self.excluded_ids = pd.concat(
                            [self.excluded_ids, excluded_ids], ignore_index=True
                        )

            df = self._create_values_for_interpolation(df)

            # Fill NaN values with interpolation
            df['loadshape'] = (
                df.groupby("id")['loadshape']
                .apply(lambda x: x.interpolate(method="linear", limit_direction="both"))
                .reset_index(drop=True)
            )

        else:
            if self.settings.TIME_PERIOD is None:
                # for loadshape type dataframe
                invalid_ids = unique_time_counts_per_id[
                    unique_time_counts_per_id < unique_time_counts
                ].index.tolist()
                invalid_ids_df = pd.DataFrame(
                    {
                        "id": invalid_ids,
                        "reason": "Unique time counts per id don't have the minimum time counts required",
                    }
                )
                self.excluded_ids = pd.concat(
                    [self.excluded_ids, invalid_ids_df], ignore_index=True
                )

            else:
                # throw out id with null values and record them, do not throw error

                # get a list of any rows with missing values
                excluded_ids = df[df.isnull().any(axis=1)]["id"].values
                if excluded_ids.size > 0:
                    excluded_ids = pd.DataFrame({"id": excluded_ids})
                    excluded_ids["reason"] = "null values in features_df"
                    self.excluded_ids = pd.concat([self.excluded_ids, excluded_ids])

        df = df[ ~df["id"].isin(self.excluded_ids["id"])]

        # pivot the loadshape_df to have the time as columns
        df = df.pivot(index="id", columns=["time"], values="loadshape")

        # Convert multi level index to single level
        df = (
            df.rename_axis(None, axis=1)
            .reset_index()
            .set_index("id")
            .drop(columns="index", axis=1, errors="ignore")
        )

        return df

    def _validate_format_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check columns missing in features_df
        if "id" not in df.columns:
            raise ValueError(f"Missing columns in features_df: 'id'")

        # get a list of any rows with missing values
        excluded_ids = df[df.isnull().any(axis=1)]["id"].values
        if excluded_ids.size > 0:
            excluded_ids = pd.DataFrame({"id": excluded_ids})
            excluded_ids["reason"] = "null values in features_df"
            self.excluded_ids = pd.concat([self.excluded_ids, excluded_ids])

        # remove any rows with missing values
        df = df.dropna()

        df.drop_duplicates(keep="first" , inplace = True)

        # drop any ids that are in excluded_ids from loadshape (or init)
        df = df[~df["id"].isin(self.excluded_ids["id"])]
        df = (
            df.reset_index()
            .set_index("id")
            .drop(columns="index", axis=1, errors="ignore")
        )

        return df

    def _convert_timeseries_to_loadshape(
        self, time_series_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Arguments:
            Time series dataframe with columns = [id, datetime, observed, observed_error, modeled, modeled_error

        Returns :
            Loadshape dataframe with columns = [id, time, loadshape]
        """

        base_df = time_series_df.copy()  # don't change the original dataframe

        # Check columns missing in time_series_df
        df_type = self.settings.LOADSHAPE_TYPE
        expected_columns = ["id", "datetime"]
        if df_type == "error":
            expected_columns.extend(["observed", "modeled"])
        else:
            expected_columns.append(df_type)

        missing_columns = [c for c in expected_columns if c not in base_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in time_series_df: {missing_columns}")

        # Check that the datetime column is actually of type datetime
        if base_df["datetime"].dtypes in _const.datetime_types:
            base_df["datetime"] = pd.to_datetime(base_df["datetime"], utc=True)
        else:
            raise ValueError("The 'datetime' column must be of datetime type")

        if df_type == "error":
            base_df["error"] = 1 - base_df["observed"] / base_df["modeled"]

        # Remove duplicates
        subset_columns = expected_columns[:-1]

        # To eliminate duplicates, sort the values by loadshape and the keep the first (i.e. the lowest) value
        base_df = base_df.sort_values(by=df_type, key=abs).drop_duplicates(subset=subset_columns, keep="first")

        base_df = self._add_index_columns_from_datetime(base_df) # Add month / day_of_week / hour / etc columns

        # Check that each id has a minimum granularity lower than requested time period, otherwise we cannot aggregate
        # get minimum time interval per id
        base_df["time_diff"] = base_df.groupby("id")["datetime"].diff()
        min_time_diff_per_id = base_df.groupby("id")["time_diff"].min() / np.timedelta64(1, 'm')

        # Get the ids that have a higher minimum granularity than defined
        if self.settings.TIME_PERIOD != 'month':
            invalid_ids = min_time_diff_per_id[
                min_time_diff_per_id > _const.min_granularity_per_time_period[self.settings.TIME_PERIOD]
            ].index.tolist()

        else:
            # Check that every ID has 12 months available.
            unique_month_counts_per_id = base_df.groupby('id')['month'].nunique()
            invalid_ids = unique_month_counts_per_id[unique_month_counts_per_id < 12].index.tolist()

        # Remove the invalid ids from the base_df
        base_df = base_df[~base_df["id"].isin(invalid_ids)]        

        # If there are any invalid ids, add them to the excluded_ids dataframe
        if invalid_ids:
            invalid_ids_df = pd.DataFrame(
                {
                    "id": invalid_ids,
                    "reason": "Minimum time interval is more than the specified TimePeriod",
                }
            )
            self.excluded_ids = pd.concat(
                [self.excluded_ids, invalid_ids_df], ignore_index=True
            )

        # Set the index to datetime
        base_df = base_df.set_index("datetime")

        # Aggregate the input time_series based on time_period

        group_by_columns = self._find_groupby_columns()

        base_df = base_df.groupby(group_by_columns)[self.settings.LOADSHAPE_TYPE]

        base_df = base_df.agg(loadshape=self.settings.AGG_TYPE).reset_index()

        # Sort the values so that the ordering is maintained correctly
        base_df = base_df.sort_values(by=group_by_columns)

        # Create the count of the index per ID
        base_df["time"] = base_df.groupby("id").cumcount() + 1

        # Validate that all the values are correct
        loadshape_df = self._validate_format_loadshape(base_df)

        return loadshape_df

    def set_data(
        self, loadshape_df=None, time_series_df=None, features_df=None
    ) -> None:
        """
            Loadshape, timeseries and features dataframes are input. The loadshape and features dataframes are validated and formatted.
            The timeseries dataframe is converted to a loadshape dataframe and then validated and formatted.

            Either loadshape or timeseries data is allowed, but not both. Atleast one of them must be provided as well.
            Features is independent of the loadshape and timeseries dataframes.

            Loadshape / timeseries only input => Clustering / IMM
            Features only input => Stratified Sampling

            Note the loadshape and features dataframe can only be set once per class.

        Args:
            Loadshape_df: columns = [id, time, loadshape]

            Time_series_df: columns = [id, datetime, observed, observed_error, modeled, modeled_error]

            Features_df: columns = [id, {feature_1}, {feature_2}, ...]

        Output:
            loadshape: index = id, columns = time, values = loadshape

            features: index = id, columns = [{feature_1}, {feature_2}, ...]


        """

        # TODO : we should accept loadshapes in unstacked version as well
        if loadshape_df is None and time_series_df is None and features_df is None:
            raise ValueError(
                "A loadshape, time series, or features dataframe must be provided."
            )

        elif loadshape_df is not None and time_series_df is not None:
            raise ValueError(
                "Both loadshape dataframe and time series dataframe are provided. Please provide only one."
            )

        if self.settings.TIME_PERIOD is not None and (loadshape_df is not None or time_series_df is None):
            # Time period should only be set if a time series dataframe is provided
            raise ValueError(
                "Time period is set, but no time series dataframe is provided. Please provide a time series dataframe."
            )

        if loadshape_df is not None:
            if self.loadshape is not None :
                raise ValueError("Loadshape Data has already been set.")
            elif self.settings.LOADSHAPE_TYPE is not None:
                raise ValueError("Loadshape Type cannot be set for a loadshape dataframe.")

            loadshape_df = self._validate_format_loadshape(loadshape_df)

        elif time_series_df is not None:
            if self.loadshape is not None:
                raise ValueError("Loadshape Data has already been set.")

            loadshape_df = self._convert_timeseries_to_loadshape(time_series_df)

        if features_df is not None:
            if self.features is not None:
                raise ValueError("Features Data has already been set.")
            features_df = self._validate_format_features(features_df)

        if loadshape_df is not None:
            # drop any ids that are in the excluded_ids list
            loadshape_df = loadshape_df[
                ~loadshape_df.index.isin(self.excluded_ids["id"])
            ]

        # If the dataframes are empty return None, not an empty dataframe
        if features_df is not None:
            self.features = features_df if not features_df.empty else None
        self.loadshape = loadshape_df if not loadshape_df.empty else None

        return self


if __name__ == "__main__":
    # Create a testing dataframe having an id, datetime of 15 min intervals, observed and modeled values
    num_intervals = 4 * 24 * 365  # 4 intervals/hour * 24 hours/day * 365 days

    # Create a DataFrame with 'id', 'datetime', 'observed', and 'modeled' columns
    df = pd.DataFrame(
        {
            "id": np.repeat(
                ["id1", "id2", "id3"], num_intervals
            ),  # only 3 ids for easier comparison
            "datetime": pd.date_range(
                start="2023-01-01", periods=num_intervals, freq="15T"
            ).tolist()
            * 3,
            "observed": np.random.rand(num_intervals * 3),  # randomized
            "modeled": np.random.rand(num_intervals * 3),  # randomized
        }
    )

    # # Create a boolean mask for Mondays and Wednesdays , will give ValueError at 80% threshold
    day_mask = df['datetime'].dt.dayofweek.isin([2])

    # # Set 'observed' and 'modeled' values to NaN for all Mondays and Wednesdays
    # df.loc[day_mask, ['observed', 'modeled']] = np.nan

    # # Remove all rows with NaN values
    df = df.loc[~day_mask]

    # Convert 'datetime' column to datetime type
    df["datetime"] = pd.to_datetime(df["datetime"])

    settings = Data_Settings(TIME_PERIOD=_const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK)

    data = Data(None).set_data(time_series_df=df)
    start_time = time.time()
    data = Data(settings).set_data(time_series_df=df)
    end_time = time.time()
    print(data.loadshape)

    execution_time = end_time - start_time
    print(f"The command took {execution_time} seconds to execute.")
