from __future__ import annotations

import numpy as np
import pandas as pd

from gridmeter._utils.base import Comparison_Group_Algorithm

from gridmeter._individual_meter_matching.settings import Settings
from gridmeter._individual_meter_matching.distance_calc_selection import (
    DistanceMatching,
)


# TODO: Should treatment, distance, and duplicated be somewhere else?
class Individual_Meter_Matching(Comparison_Group_Algorithm):
    def __init__(self, settings: Settings | None):
        if settings is None:
            self.settings = Settings()
        else:
            self.settings = settings

        self.dist_metric = settings.distance_metric
        if self.dist_metric == "manhattan":
            self.dist_metric = "cityblock"

    def get_comparison_group(self, df_ls_t, df_ls_cp, weights=None):
        distance_matching = DistanceMatching(self.settings)

        # Get clusters
        clusters = distance_matching.get_comparison_group(
            df_ls_t, df_ls_cp, weights=weights
        )
        clusters["cluster"] = 0
        clusters["weight"] = 1.0
        clusters = clusters.reset_index().set_index("id")

        # reorder columns
        clusters = clusters[
            ["treatment", "distance", "duplicated", "cluster", "weight"]
        ]

        # Create treatment_weights
        t_ids = df_ls_t.index.unique()
        coeffs = np.ones(t_ids.values.size)

        treatment_weights = pd.DataFrame(coeffs, index=t_ids, columns=["pct_cluster_0"])
        treatment_weights.index.name = "id"

        self.clusters = clusters
        self.treatment_weights = treatment_weights

        return clusters, treatment_weights
