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
import numpy as np
import pandas as pd
from eemeter.eemeter.models.daily.model import DailyModel
from eemeter.eemeter.models.daily.data import DailyBaselineData
from eemeter.eemeter.models.daily.optimize_results import OptimizedResult


class TestFitModel:
    @classmethod
    def setup_class(cls):
        # Create a sample meter data DataFrame from the test data
        df = pd.read_csv("tests/daily_model/test_data.csv")
        df.index = pd.to_datetime(df["datetime"])
        df = df[["temperature", "observed"]]
        cls.meter_data = DailyBaselineData(df, is_electricity_data=True)

    def test_fit_model(self):
        # Create a DailyModel instance
        fm = DailyModel().fit(self.meter_data, ignore_disqualification=True)

        # Test that the combinations attribute is a list
        assert isinstance(fm.combinations, list)

        # Test that the combinations attribute is as expected
        expected_combinations = [
            "fw-su_sh_wi",
            "fw-sh_wi__fw-su",
            "fw-sh_wi__wd-su__we-su",
            "wd-su_sh_wi__we-su_sh_wi",
            "fw-su__wd-sh_wi__we-sh_wi",
            "wd-su__wd-sh_wi__we-su_sh_wi",
            "wd-su_sh_wi__we-su__we-sh_wi",
            "wd-su__wd-sh_wi__we-su__we-sh_wi",
        ]
        assert fm.combinations == expected_combinations

        # Test that the components attribute is a list
        assert isinstance(fm.components, list)

        # Test that the components attribute is as expected
        expected_components = [
            "fw-su",
            "wd-su",
            "we-su",
            "fw-sh_wi",
            "wd-sh_wi",
            "we-sh_wi",
            "fw-su_sh_wi",
            "wd-su_sh_wi",
            "we-su_sh_wi",
        ]
        assert fm.components == expected_components

        # Test that the fit_components attribute is a dictionary
        assert isinstance(fm.fit_components, dict)

        # Test that the wRMSE_base attribute is a float
        assert isinstance(fm.wRMSE_base, float)
        assert np.isclose(fm.wRMSE_base, 18.389335982383994)

        # Test that the best combination is as expected
        expected_best_combination = "wd-su_sh_wi__we-su_sh_wi"
        assert fm.best_combination == expected_best_combination

        # Test that the final model is as expected
        combinations = expected_best_combination.split("__")
        for combination in combinations:
            assert isinstance(fm.model[combination], OptimizedResult)

        # Test that the error attribute values are as expected
        expected_model_error = {
            "wRMSE": 16.95324536039207,
            "RMSE": 16.95324536039207,
            "MAE": 13.38096518529209,
            "CVRMSE": 0.32064123575928577,
            "PNRMSE": 0.270778281497731,
        }
        assert fm.error == expected_model_error
