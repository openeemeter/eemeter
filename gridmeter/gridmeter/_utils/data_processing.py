from gridmeter._utils.loadshape_settings import Data_Settings
import pandas as pd

# TODO: Should this class go in const.py, here, or data_settings.py?
class DataConstants:
    """
        Utility class defining the constants used by the Data class.
    """


    time_periods = [
        "hourly",
        "month",
        "hourly_month",
        "day_of_week",
        "hourly_day_of_week",
        "weekday_weekend",
        "hourly_weekday_weekend",
        "season_day_of_week",
        "season_hourly_day_of_week",
        "season_weekday_weekend",
        "season_hourly_weekday_weekend",
    ]

    time_period_row_counts = {
        "hourly": 24,
        "month": 12,
        "hourly_month": 24 * 12,
        "day_of_week": 7,
        "hourly_day_of_week": 24 * 7,
        "weekday_weekend": 2,
        "hourly_weekday_weekend": 24 * 2,
        "season_day_of_week": 3 * 7,
        "season_hourly_day_of_week": 3 * 24 * 7,
        "season_weekday_weekend": 3 * 2,
        "season_hourly_weekday_weekend": 3 * 24 * 2,
    }

    # This list ordering is important for the groupby columns
    unique_time_periods = ["season", "month", "day_of_week", "weekday_weekend", "hour"]


class Data:
    def __init__(self, settings: Data_Settings | None  = None):
        if settings is None:
            self.settings = Data_Settings()

        self.excluded_ids = pd.DataFrame(columns=["id", "reason"])

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

        for period in DataConstants.unique_time_periods:
            if period in self.settings.TIME_PERIOD:
                cols.append(period)

        return cols

    def _add_index_columns_from_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add hour column
        if "hour" in self.settings.TIME_PERIOD:
            df["hour"] = df.index.hour

        # Add month column
        if "month" in self.settings.TIME_PERIOD:
            df["month"] = df.index.month

        # Add day_of_week column
        if "day_of_week" in self.settings.TIME_PERIOD:
            df["day_of_week"] = df.index.dayofweek

        # Add weekday_weekend column
        if "weekday_weekend" in self.settings.TIME_PERIOD:
            df["weekday_weekend"] = df.index.dayofweek

            # Setting the ordering to weekday, weekend
            df["weekday_weekend"] = (
                df["weekday_weekend"]
                .map(self.settings.WEEKDAY_WEEKEND._NUM_DICT)
                .map(self.settings.WEEKDAY_WEEKEND._ORDER)
            )

        # Add season column
        if "season" in self.settings.TIME_PERIOD:
            df["season"] = df.index.month.map(self.settings.SEASON._NUM_DICT).map(
                self.settings.SEASON._ORDER
            )

        return df

    def _validate_format_loadshape(self, df: pd.DataFrame) -> pd.DataFrame:
        # Check columns missing in loadshape_df
        expected_columns = ["id", "time", "loadshape"]
        missing_columns = [c for c in expected_columns if c not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing columns in time_series_df: {missing_columns}")

        # Check if all values are present in the columns as required
        # Else update the values via interpolation if missing, also ignore duplicates if present

        # loadshape df has the "time" column, whereas timeseries df has the "datetime" column
        subset_columns = expected_columns[:-1]
        
        # TODO: What to do with duplicates?
        df = df.drop_duplicates(subset=subset_columns, keep="first")

        if self.settings.INTERPOLATE_MISSING:
            # Check that the number of missing values is less than the threshold
            for id, group in df.groupby("id"):
                if (
                    group.count().min()
                    < self.settings.MIN_DATA_PCT_REQUIRED
                    * DataConstants.time_period_row_counts[self.settings.TIME_PERIOD]
                ):
                    raise ValueError(
                        f"Missing minimum threshold number of values in dataframe for id: {id}"
                    )

            # Fill NaN values with interpolation
            df = (
                df.groupby("id")
                .apply(lambda x: x.interpolate(method="linear", limit_direction="both"))
                .reset_index(drop=True)
            )

            # TODO : Interpolation should only occur on within seasons, not across seasons

        else:
            # TODO: throw out meters with null values and record them, do not throw error
            missing_values = df[df.isnull().any(axis=1)]
            if missing_values.shape[0] > 0:
                raise ValueError(
                    f"Missing values in loadshape_df: {missing_values.shape[0]}"
                )
            
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
            raise ValueError(f"Missing columns in time_series_df: 'id'")
        
        # get a list of any rows with missing values
        excluded_ids = df[df.isnull().any(axis=1)]['id'].values
        if excluded_ids.size > 0:
            excluded_ids = pd.DataFrame({"id": excluded_ids})
            excluded_ids["reason"] = "null values in features_df"
            self.excluded_ids = pd.concat([self.excluded_ids, excluded_ids])

        # remove any rows with missing values
        df = df.dropna()

        # TODO: What to do with duplicates?
        df = df.drop_duplicates(keep="first")

        # drop any ids that are in excluded_ids from loadshape (or init)
        df = df[~df['id'].isin(self.excluded_ids['id'])]
        df = df.reset_index().set_index("id").drop(columns="index", axis=1, errors="ignore")

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

        # Check columns missing in time_series_df
        df_type = self.settings.LOADSHAPE_TYPE
        expected_columns = ["id", "datetime"]
        if df_type == "error":
            expected_columns.extend(["observed", "modeled"])
        else:
            expected_columns.append(df_type)

        missing_columns = [c for c in expected_columns if c not in time_series_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in time_series_df: {missing_columns}")

        # Check that the datetime column is actually of type datetime
        if time_series_df["datetime"].dtypes != "datetime64[ns]":
            raise ValueError("The 'datetime' column must be of datetime type")

        if df_type == "error":
            pass  # TODO: calculate error

        # Create a base df for adding all required columns
        base_df = time_series_df.set_index("datetime")
        base_df = self._add_index_columns_from_datetime(base_df)

        # Aggregate the input time_series based on time_period

        group_by_columns = self._find_groupby_columns()

        grouped_df = base_df.groupby(group_by_columns)[self.settings.LOADSHAPE_TYPE]

        agg_df = grouped_df.agg(loadshape=self.settings.AGG_TYPE).reset_index()

        # Sort the values so that the ordering is maintained correctly
        agg_df = agg_df.sort_values(by=group_by_columns)

        # Create the count of the index per ID
        agg_df["time"] = agg_df.groupby("id").cumcount() + 1

        # Validate that all the values are correct
        loadshape_df = self._validate_format_loadshape(agg_df)

        return loadshape_df

    def set_data(self, loadshape_df=None, time_series_df=None, features_df=None) -> None:
        """

        Args:
            Loadshape_df: columns = [id, time, loadshape]

            Time_series_df: columns = [id, datetime, observed, observed_error, modeled, modeled_error]

            Features_df: columns = [id, {feature_1}, {feature_2}, ...]

        Output:
            loadshape: index = id, columns = time, values = loadshape

            features: index = id, columns = [{feature_1}, {feature_2}, ...]


        """
        if loadshape_df is None and time_series_df is None and features_df is None:
            raise ValueError(
                "A loadshape, time series, or features dataframe must be provided."
            )

        elif loadshape_df is not None and time_series_df is not None:
            raise ValueError(
                "Both loadshape dataframe and time series dataframe are provided. Please provide only one."
            )

        if loadshape_df is not None:
            loadshape_df = self._validate_format_loadshape(loadshape_df)

        elif time_series_df is not None:
            loadshape_df = self._convert_timeseries_to_loadshape(time_series_df)            

        # TODO: need to track dropped ids in loadshape and apply to features
        if features_df is not None:
            features_df = self._validate_format_features(features_df)            

        if loadshape_df is not None:
            # drop any ids that are in the excluded_ids list
            loadshape_df = loadshape_df[~loadshape_df.index.isin(self.excluded_ids['id'])]

        self.features = features_df
        self.loadshape = loadshape_df