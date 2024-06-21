from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd

from gridmeter._utils.base_comparison_group import Comparison_Group_Algorithm

from gridmeter.random_sampling.settings import Settings


class Random_Sampling(Comparison_Group_Algorithm):
    def __init__(self, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()
        
        self.settings = settings

    def _create_clusters_df(self, df_raw):
        clusters = df_raw
        clusters["cluster"] = 0
        clusters["weight"] = 1.0
        clusters = clusters.reset_index().set_index("id")

        # reorder columns
        clusters = clusters[["cluster", "weight"]]

        return clusters
    

    def _create_treatment_weights_df(self, ids):
        coeffs = np.ones(len(ids))

        treatment_weights = pd.DataFrame(coeffs, index=ids, columns=["pct_cluster_0"])
        treatment_weights.index.name = "id"

        return treatment_weights
    

    def get_comparison_group(self, treatment_data, comparison_pool_data, weights=None):
        settings = self.settings

        if settings.N_METERS_TOTAL is not None:
            n_meters = self.settings.N_METERS_TOTAL

        elif settings.N_METERS_PER_TREATMENT is not None:
            n_treatment_meters = len(treatment_data.ids)
            n_meters = n_treatment_meters * settings.N_METERS_PER_TREATMENT

        else:
            raise ValueError("N_METERS_TOTAL or N_METERS_PER_TREATMENT must be defined")
        
        self.treatment_data = treatment_data
        self.comparison_pool_data = comparison_pool_data

        self.treatment_ids = treatment_data.ids
        self.treatment_loadshape = treatment_data.loadshape
        self.comparison_pool_loadshape = comparison_pool_data.loadshape

        # randomly sample n_meters from comparison pool
        df_cg = comparison_pool_data.loadshape.sample(n_meters, random_state=settings.SEED)

        clusters = self._create_clusters_df(df_cg)

        # Create treatment_weights
        treatment_weights = self._create_treatment_weights_df(self.treatment_ids)

        # Assign dfs to self
        self.clusters = clusters
        self.treatment_weights = treatment_weights

        return clusters, treatment_weights
