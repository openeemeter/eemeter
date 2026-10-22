#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import pytest

from opendsm.eemeter import HourlyCaltrackBaselineData, HourlyCaltrackModel



@pytest.fixture(scope="session")
def caltrack_hourly_model(comstock_hourly):
    """The hourly CalTRACK model fit once on the ComStock baseline meter; tests read it and
    never mutate it."""
    df_b, _ = comstock_hourly
    meter = df_b[["observed"]].rename(columns={"observed": "value"}).copy()
    baseline = HourlyCaltrackBaselineData.from_series(
        meter, df_b["temperature"], is_electricity_data=True
    )
    model = HourlyCaltrackModel().fit(baseline)

    return model
