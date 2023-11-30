"""
main module to import and to expose external api
"""

from __future__ import annotations

import pandas as pd

from gridmeter._utils.base_comparison_group import Comparison_Group_Algorithm
from gridmeter._clustering import settings as _settings, cluster as _cluster


# TODO: Make this work better with Data class
# TODO: get_clusters and match_treatment_to_cluster will break if passed a Data class
class Clustering(Comparison_Group_Algorithm):
    Cluster = None

    def __init__(self, settings: _settings.Settings | None = None):
        if settings is None:
            settings = _settings.Settings()

        self.settings = settings

    def get_clusters(self, df_ls_cp: pd.DataFrame):
        self.Cluster = (
            _cluster.ClusterResult.from_comparison_pool_loadshapes_and_settings(
                df_cp_ls=df_ls_cp, s=self.settings
            )
        )
        self.clusters = self.Cluster.cluster_df

        return self.clusters

    def match_treatment_to_clusters(self, df_ls_t: pd.DataFrame):
        if self.Cluster is None:
            raise ValueError(
                "Comparison group has been not been clustered. Please run 'get_clusters' first."
            )

        self.treatment_weights = self.Cluster.get_match_treatment_to_cluster_df(
            treatment_loadshape_df=df_ls_t
        )

        return self.treatment_weights

    def get_comparison_group(self, treatment_data, comparison_pool_data):
        df_t = treatment_data.loadshape
        df_cp = comparison_pool_data.loadshape

        # TODO: Should fix in clustering algorithm to eliminate need for this transformation
        df_t = df_t.stack().reset_index().rename(columns={"level_1": "time", 0: "ls"}).set_index(["id", "time"])
        df_cp = df_cp.stack().reset_index().rename(columns={"level_1": "time", 0: "ls"}).set_index(["id", "time"])

        clusters = self.get_clusters(df_cp)
        treatment_weights = self.match_treatment_to_clusters(df_t)

        return clusters, treatment_weights
