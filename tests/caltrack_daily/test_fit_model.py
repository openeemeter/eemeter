import numpy as np
import pandas as pd
from eemeter.models import DailyModel
from eemeter.caltrack.daily.optimize_results import OptimizedResult


class TestFitModel:
    @classmethod
    def setup_class(cls):
        # Create a sample meter data DataFrame from the test data
        cls.meter_data = pd.read_csv("tests/caltrack_daily/test_data.csv")

    def test_fit_model(self):
        # Create a DailyModel instance
        fm = DailyModel().fit(self.meter_data)

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

        # Test that the error attribute is a dictionary with the expected keys
        assert isinstance(fm.error, dict)
        assert set(fm.error.keys()) == set(
            [
                "wRMSE_train",
                "wRMSE_test",
                "RMSE_train",
                "RMSE_test",
                "MAE_train",
                "MAE_test",
                "CVRMSE_train",
            ]
        )

        # Test that the error attribute values are as expected
        expected_model_error = {
            "wRMSE_train": 16.95324536039207,
            "wRMSE_test": np.nan,
            "RMSE_train": 16.95324536039207,
            "RMSE_test": np.nan,
            "MAE_train": 13.38096518529209,
            "MAE_test": np.nan,
            "CVRMSE_train": 0.32064123575928577,
        }
        assert fm.error == expected_model_error
