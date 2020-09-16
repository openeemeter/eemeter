import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def col_name():
    return 'col1'


@pytest.fixture
def df_train(col_name):
    return pd.DataFrame(
        [
            {"id": f"id_train_{x}", col_name: x}
            for x in (
                list(np.arange(0, 2, 0.1))
                + list(np.arange(2, 4, 0.5))
                + list(np.arange(4, 6, 1))
                + list(np.arange(6, 10, 0.2))
            )
        ]
    )


@pytest.fixture
def df_test(col_name):
    return pd.DataFrame(
        [
            {"id": f"id_test_{x}", col_name: x}
            for x in np.arange(0, 20, 0.01)
        ]
    )



@pytest.fixture
def df_equiv(df_train, df_test):
    df_train_records = pd.DataFrame(
        [
            {
                "id": dim_project_site_meter_id,
                "month": month,
                "baseline_predicted_usage": month*i,
            }
            for month in range(1, 13)
            for i, dim_project_site_meter_id in enumerate(df_train["id"].values)
        ]
    )
    df_test_records = pd.DataFrame(
        [
            {
                "id": dim_project_site_meter_id,
                "month": month,
                "baseline_predicted_usage": (13 - month) * i * 0.1,
            }
            for month in range(1, 13)
            for i, dim_project_site_meter_id in enumerate(df_test["id"].values)
        ]
    )
    return pd.concat([df_train_records, df_test_records])
