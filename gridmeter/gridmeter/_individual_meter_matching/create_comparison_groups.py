from __future__ import annotations

import numpy as np
import pandas as pd

from gridmeter._utils.base_comparison_group import Comparison_Group_Algorithm

from gridmeter._individual_meter_matching.settings import Settings
from gridmeter._individual_meter_matching.distance_calc_selection import (
    DistanceMatching,
)


# TODO: Should treatment, distance, and duplicated be somewhere else?
class Individual_Meter_Matching(Comparison_Group_Algorithm):
    def __init__(self, settings: Settings | None = None):
        if settings is None:
            self.settings = Settings()
        
        self.settings = settings

        self.dist_metric = settings.distance_metric
        if self.dist_metric == "manhattan":
            self.dist_metric = "cityblock"

    def _create_clusters_df(self, df_raw):
        clusters = df_raw
        clusters["cluster"] = 0
        clusters["weight"] = 1.0
        clusters = clusters.reset_index().set_index("id")

        # reorder columns
        clusters = clusters[
            ["treatment", "distance", "duplicated", "cluster", "weight"]
        ]

        return clusters
    

    def _create_treatment_weights_df(self, ids):
        coeffs = np.ones(ids.values.size)

        treatment_weights = pd.DataFrame(coeffs, index=ids, columns=["pct_cluster_0"])
        treatment_weights.index.name = "id"

        return treatment_weights
    

    def get_comparison_group(self, df_ls_t, df_ls_cp, weights=None):
        distance_matching = DistanceMatching(self.settings)

        # Get clusters
        df_raw = distance_matching.get_comparison_group(
            df_ls_t, df_ls_cp, weights=weights
        )
        clusters = self._create_clusters_df(df_raw)

        # Create treatment_weights
        t_ids = df_ls_t.index.unique()
        treatment_weights = self._create_treatment_weights_df(t_ids)

        # Assign dfs to self
        self.clusters = clusters
        self.treatment_weights = treatment_weights

        return clusters, treatment_weights
