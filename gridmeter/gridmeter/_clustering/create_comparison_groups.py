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

    def get_comparison_group(self, df_ls_t: pd.DataFrame, df_ls_cp: pd.DataFrame):
        df_ls_t = df_ls_t.stack().reset_index().rename(columns={0: "ls"})
        df_ls_cp = df_ls_cp.stack().reset_index().rename(columns={0: "ls"})

        clusters = self.get_clusters(df_ls_cp)
        treatment_weights = self.match_treatment_to_clusters(df_ls_t)

        return clusters, treatment_weights
