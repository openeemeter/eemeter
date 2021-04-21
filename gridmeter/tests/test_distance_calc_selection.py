from gridmeter.distance_calc_selection import distance_match
import pandas as pd
import random


def test_distance_match():
    random.seed(1)

    treatment_group = pd.DataFrame(
        [
            {
                "id": f"t_{i}",
                "month_1": random.random(),
                "month_2": random.random(),
                "month_3": random.random(),
            }
            for i in range(1, 10)
        ]
    ).set_index("id")
    comparison_pool = pd.DataFrame(
        [
            {
                "id": f"c_{i}",
                "month_1": random.random(),
                "month_2": random.random(),
                "month_3": random.random(),
            }
            for i in range(1, 100)
        ]
    ).set_index("id")

    comparison_group = distance_match(treatment_group, comparison_pool)
    assert not comparison_group.empty
