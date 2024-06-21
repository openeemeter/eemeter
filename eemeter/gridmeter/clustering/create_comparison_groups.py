"""
main module to import and to expose external api
"""

from __future__ import annotations
from typing import Optional

import pandas as pd

from gridmeter._utils.base_comparison_group import Comparison_Group_Algorithm
from gridmeter.clustering import settings as _settings, cluster as _cluster


class Clustering(Comparison_Group_Algorithm):
    Cluster = None
    comparison_pool_loadshape = None
    treatment_loadshape = None

    def __init__(self, settings: Optional[_settings.Settings] = None):
        if settings is None:
            settings = _settings.Settings()

        self.settings = settings

    def get_clusters(self, comparison_pool_data):
        self.comparison_pool_data = comparison_pool_data
        self.comparison_pool_loadshape = comparison_pool_data.loadshape
        
        self.Cluster = (
            _cluster.ClusterResult.from_comparison_pool_loadshapes_and_settings(
                df_cp=self.comparison_pool_loadshape, 
                s=self.settings
            )
        )
        self.clusters = self.Cluster.cluster_df

        return self.clusters

    def match_treatment_to_clusters(self, treatment_data):
        if self.Cluster is None:
            raise ValueError(
                "Comparison group has been not been clustered. Please run 'get_clusters' first."
            )
        
        self.treatment_data = treatment_data
        self.treatment_ids = treatment_data.ids
        self.treatment_loadshape = treatment_data.loadshape

        self.treatment_weights = self.Cluster.get_match_treatment_to_cluster_df(
            treatment_loadshape_df=self.treatment_loadshape,
            s=self.settings
        )

        return self.treatment_weights

    def get_comparison_group(self, treatment_data, comparison_pool_data):
        clusters = self.get_clusters(comparison_pool_data)
        treatment_weights = self.match_treatment_to_clusters(treatment_data)

        return clusters, treatment_weights
