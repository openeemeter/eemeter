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

from pathlib import Path
import copy
from typing import Optional, Union

from eemeter.eemeter.common.data_processor_utilities import (
    compute_minimum_granularity,
    remove_duplicates,
)
from eemeter.common.hourly_interpolation import interpolate
from eemeter.eemeter.common.features import compute_temperature_features
from eemeter.eemeter.common.sufficiency_criteria import HourlySufficiencyCriteria
from eemeter.eemeter.common.warnings import EEMeterWarning

import numpy as np
import pandas as pd


# TODO move to settings/const
_MAX_MISSING_HOURS_PCT = 10


class NREL_Weather_API:  # TODO: reload data for all years
    api_key = (
        "---"  # get your own key from https://developer.nrel.gov/signup/  #Required
    )
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


class _HourlyData:
    """Private base class for hourly baseline and reporting data

    Will raise exception during data sufficiency check if instantiated
    """

    # TODO do we need to specify elec? consider a default true? how common are hourly gas meters
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool, **kwargs: dict):
        self._df = None
        self.warnings = []
        self.disqualification = []
        self.is_electricity_data = is_electricity_data
        self.tz = None

        # TODO copied from HourlyData
        self.to_be_interpolated_columns = []
        self.interp = None
        self.outputs = []
        self.pv_start = None

        self.kwargs = copy.deepcopy(kwargs)
        if "outputs" in self.kwargs:
            self.outputs = copy.deepcopy(self.kwargs["outputs"])
        else:
            self.outputs = ["temperature", "observed"]

        self.missing_values_amount = {}
        self.too_many_missing_data = False

        self._df = self._set_data(df)

        # sufficiency_df = self._df.merge(
        #     temp_coverage, left_index=True, right_index=True, how="left"
        # )
        # disqualification, warnings = self._check_data_sufficiency(sufficiency_df)
        disqualification, warnings = [], []

        self.disqualification += disqualification
        self.warnings += warnings
        self.log_warnings()

    @property
    def df(self):
        """
        Get the corrected input data stored in the class. The actual dataframe is immutable, this returns a copy.

        Returns
        -------
            pandas.DataFrame or None: A copy of the DataFrame if it exists, otherwise None.
        """
        if self._df is None:
            return None
        else:
            return self._df.copy()

    @classmethod
    def from_series(
        cls,
        meter_data: Union[pd.Series, pd.DataFrame],
        temperature_data: Union[pd.Series, pd.DataFrame],
        is_electricity_data,
    ):
        raise NotImplementedError(
            "Unimplemented until full release--use regular constructor with complete hourly dataframe."
        )

    def log_warnings(self):
        """
        Logs the warnings and disqualifications associated with the data.
        """
        for warning in self.warnings + self.disqualification:
            warning.warn()

    def _get_contiguous_datetime(self, df):
        # get earliest datetime and latest datetime
        # make earliest start at 0 and latest end at 23, this ensures full days
        earliest_datetime = df.index.min().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        latest_datetime = df.index.max().replace(
            hour=23, minute=0, second=0, microsecond=0
        )

        # create a new index with all the hours between the earliest and latest datetime
        complete_dt = pd.date_range(
            start=earliest_datetime, end=latest_datetime, freq="H"
        )

        # merge meter data with complete_dt
        df = df.reindex(complete_dt)

        df["date"] = df.index.date
        df["hour_of_day"] = df.index.hour

        return df

    # TODO move to common/transforms rather than operating on self
    def _interpolate(self, df):
        # make column of interpolated boolean if any observed or temperature is nan
        # check if in each row of the columns in output has nan values, the interpolated column will be true
        if "to_be_interpolated_columns" in self.kwargs:
            self.to_be_interpolated_columns = self.kwargs[
                "to_be_interpolated_columns"
            ].copy()
            self.outputs += [
                f"{col}"
                for col in self.to_be_interpolated_columns
                if col not in self.outputs
            ]
        else:
            self.to_be_interpolated_columns = ["temperature", "observed"]
            if "ghi" in df.columns:
                self.to_be_interpolated_columns.append("ghi")

        # for col in self.outputs:
        #     if col not in self.to_be_interpolated_columns: #TODO: this might be diffrent for supplemental data
        #         self.to_be_interpolated_columns += [col]

        # #TODO: remove this in the actual implementation, this is just for CalTRACK testing
        # if 'model' in self.outputs:
        #     self.to_be_interpolated_columns += ['model']

        for col in self.to_be_interpolated_columns:
            if f"interpolated_{col}" in df.columns:
                continue
            self.outputs += [f"interpolated_{col}"]

        # check how many nans are in the columns
        nan_numbers_cols = df[self.to_be_interpolated_columns].isna().sum()
        # if the number of nan is more than max_missing_hours_pct, then we we flag them
        # TODO: this should be as a part of disqualification and warning/error logs
        for col in self.to_be_interpolated_columns:
            if nan_numbers_cols[col] > len(df) * _MAX_MISSING_HOURS_PCT / 100:
                if not self.too_many_missing_data:
                    self.too_many_missing_data = True
                self.missing_values_amount[col] = nan_numbers_cols[col]

        # we can add kwargs to the interpolation class like: inter_kwargs = {"n_cor_idx": self.kwargs["n_cor_idx"]}
        df = interpolate(df, columns=self.to_be_interpolated_columns)

        return df

    def _add_pv_start_date(self, df, model_type="TS"):
        # add pv start date here to avoid interpolating the pv start date
        if "metadata" in self.kwargs:
            if "pv_start" in self.kwargs["metadata"]:
                self.pv_start = self.kwargs["metadata"]["pv_start"]
                if self.pv_start is not None:
                    self.pv_start = pd.to_datetime(self.pv_start).date()
        if self.pv_start is None:
            self.pv_start = df.index.date.min()

        if "ts" in model_type.lower() or "time" in model_type.lower():
            df["has_pv"] = 0
            df.loc[df["date"] >= self.pv_start, "has_pv"] = 1

        else:
            df["has_pv"] = False
            df.loc[df["date"] >= self.pv_start, "has_pv"] = True
        return df

    def _merge_meter_temp(self, meter, temp):
        df = meter.merge(
            temp, left_index=True, right_index=True, how="left"
        ).tz_convert(meter.index.tz)
        return df

    def _check_data_sufficiency(self, sufficiency_df):
        raise NotImplementedError(
            "Can't instantiate class _HourlyData, use HourlyBaselineData or HourlyReportingData."
        )

    def _set_data(self, data: pd.DataFrame):
        df = data.copy()
        expected_columns = [
            "observed",
            "temperature",
            # "ghi",
        ]
        # TODO maybe check datatypes
        if not set(expected_columns).issubset(set(df.columns)):
            # show the columns that are missing
            raise ValueError(
                "Data is missing required columns: {}".format(
                    set(expected_columns) - set(df.columns)
                )
            )

        # Check that the datetime index is timezone aware timestamp
        if not isinstance(df.index, pd.DatetimeIndex) and "datetime" not in df.columns:
            raise ValueError("Index is not datetime and datetime not provided")

        elif "datetime" in df.columns:
            if df["datetime"].dt.tz is None:
                raise ValueError("Datatime is missing timezone information")
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

        elif df.index.tz is None:
            raise ValueError("Datatime is missing timezone information")
        elif str(df.index.tz) == "UTC":
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.data_quality.utc_index",
                    description=(
                        "Datetime index is in UTC. Use tz_localize() with the local timezone to ensure correct aggregations"
                    ),
                    data={},
                )
            )
        self.tz = df.index.tz

        # prevent later issues when merging on generated datetimes, which default to ns precision
        # there is almost certainly a smoother way to accomplish this conversion, but this works
        if df.index.dtype.unit != "ns":
            utc_index = df.index.tz_convert("UTC")
            ns_index = utc_index.astype("datetime64[ns, UTC]")
            df.index = ns_index.tz_convert(self.tz)

        # Convert electricity data having 0 meter values to NaNs
        if self.is_electricity_data:
            df.loc[df["observed"] == 0, "observed"] = np.nan

        # Caltrack 2.3.2 - Drop duplicates
        df = remove_duplicates(df)

        df = self._get_contiguous_datetime(df)
        df = self._interpolate(df)
        df = self._add_pv_start_date(df)

        return df


class HourlyBaselineData(_HourlyData):
    def _check_data_sufficiency(self, sufficiency_df):
        hsc = HourlySufficiencyCriteria(
            data=sufficiency_df, is_electricity_data=self.is_electricity_data
        )
        hsc.check_sufficiency_baseline()
        disqualification = hsc.disqualification
        warnings = hsc.warnings

        return disqualification, warnings


class HourlyReportingData(_HourlyData):
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool, **kwargs: dict):
        df = df.copy()
        if "observed" not in df.columns:
            df["observed"] = np.nan

        super().__init__(df, is_electricity_data, **kwargs)

    def _check_data_sufficiency(self, sufficiency_df):
        hsc = HourlySufficiencyCriteria(
            data=sufficiency_df, is_electricity_data=self.is_electricity_data
        )
        hsc.check_sufficiency_reporting()
        disqualification = hsc.disqualification
        warnings = hsc.warnings

        return disqualification, warnings
