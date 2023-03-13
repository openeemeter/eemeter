#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2023 OpenEEmeter contributors

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
import pandas as pd
from eemeter.processing import *


def test_add_freq(il_electricity_cdd_hdd_hourly):
    meter_data = il_electricity_cdd_hdd_hourly["meter_data"]

    # make DateTimeIndex timezone-naive
    meter_data.index = meter_data.index.tz_localize(None)

    # infer frequency
    meter_data.index = add_freq(meter_data.index)
    assert meter_data.index.freq == "H"


def test_trim_two_dataframes(
    uk_electricity_hdd_only_hourly_sample_1, uk_electricity_hdd_only_hourly_sample_2
):

    df1 = uk_electricity_hdd_only_hourly_sample_1["meter_data"]
    df2 = uk_electricity_hdd_only_hourly_sample_2["meter_data"]

    df1_trimmed, df2_trimmed = trim(df1, df2)

    assert (
        df1.index[0] == df1.index.min()
        and df2.index[0] == df2.index.min()
        and df1.index[0] != df2.index[0]
    )

    assert (
        df1.index[-1] == df1.index.max()
        and df2.index[-1] == df2.index.max()
        and df1.index[-1] != df2.index[-1]
    )

    assert df1_trimmed.index[0] == df2_trimmed.index[0]
    assert df1_trimmed.index.min() == df2_trimmed.index.min()
    assert df1_trimmed.index[-1] == df2_trimmed.index[-1]
    assert df1_trimmed.index.max() == df2_trimmed.index.max()


def test_sum_gas_and_elec(il_gas_hdd_only_hourly, il_electricity_cdd_hdd_hourly):
    gas = il_gas_hdd_only_hourly["meter_data"]
    elec = il_electricity_cdd_hdd_hourly["meter_data"]

    total = sum_gas_and_elec(gas, elec)

    for x, y, z in zip(total["value"], gas["value"], elec["value"]):
        assert x == y + z


def test_format_temperature_data_for_eemeter(il_electricity_cdd_hdd_hourly):
    temperature_data = il_electricity_cdd_hdd_hourly["temperature_data"]

    # temperature_data to pd.DateFrame
    temperature_data = pd.DataFrame(temperature_data)

    # flipping df
    temperature_data = temperature_data.reindex(index=temperature_data.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34").tz_localize("UTC")
    temperature_data.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    temperature_data.rename(columns={"value": "consumption"}, inplace=True)

    temperature_data_reformatted = format_temperature_data_for_eemeter(temperature_data)

    assert isinstance(temperature_data_reformatted, pd.Series)
    assert (
        temperature_data_reformatted.index[0] < temperature_data_reformatted.index[-1]
    )
    assert temperature_data_reformatted.index.freq == "H"
    assert temperature_data_reformatted.index.tzinfo is not None


def test_format_energy_data_for_caltrack_hourly(il_electricity_cdd_hdd_hourly):
    df = il_electricity_cdd_hdd_hourly["meter_data"]
    # flipping df
    df = df.reindex(index=df.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34").tz_localize("UTC")
    df.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    df.rename(columns={"value": "consumption"}, inplace=True)

    # df_flipped to pd.Series
    df = df.squeeze()

    df_reformatted = format_energy_data_for_eemeter(df, method="hourly")

    assert isinstance(df_reformatted, pd.DataFrame)
    assert df_reformatted.index[0] < df_reformatted.index[-1]
    assert df_reformatted.index.freq == "H"
    assert df_reformatted.columns[0] == "value"
    assert df_reformatted.index.tzinfo is not None
    assert len(df_reformatted.columns) == 1


def test_format_energy_data_for_caltrack_daily(il_electricity_cdd_hdd_daily):
    df = il_electricity_cdd_hdd_daily["meter_data"]
    # flipping df
    df = df.reindex(index=df.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34").tz_localize("UTC")
    df.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    df.rename(columns={"value": "consumption"}, inplace=True)

    # df_flipped to pd.Series
    df = df.squeeze()

    df_reformatted = format_energy_data_for_eemeter(df, method="daily")

    assert isinstance(df_reformatted, pd.DataFrame)
    assert df_reformatted.index[0] < df_reformatted.index[-1]
    assert df_reformatted.index.freq == "D"
    assert df_reformatted.columns[0] == "value"
    assert df_reformatted.index.tzinfo is not None
    assert len(df_reformatted.columns) == 1


def test_format_energy_data_for_eemeter_billing(il_electricity_cdd_hdd_daily):
    df = il_electricity_cdd_hdd_daily["meter_data"]
    # flipping df
    df = df.reindex(index=df.index[::-1])

    # inserting new value of 0.04 at 09.34 22/11/2015
    new_start = pd.to_datetime("22/11/2015 09:34").tz_localize("UTC")
    df.loc[new_start] = [0.04]

    # rename column name to 'consumption'
    df.rename(columns={"value": "consumption"}, inplace=True)

    # df_flipped to pd.Series
    df = df.squeeze()

    df_reformatted = format_energy_data_for_eemeter(df, method="billing")

    assert isinstance(df_reformatted, pd.DataFrame)
    assert df_reformatted.index[0] < df_reformatted.index[-1]
    assert df_reformatted.index.freq == "M"
    assert df_reformatted.columns[0] == "value"
    assert df_reformatted.index.tzinfo is not None
    assert len(df_reformatted.columns) == 1


# tests for caltrack_hourly and caltrack_daily NOT required as simply functionalising other eemeter functions (subejct to test).
