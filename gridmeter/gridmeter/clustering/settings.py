"""
module defines settings to be used for clustering
"""

from __future__ import annotations

import pydantic

import gridmeter.clustering.const as _const


class Settings(pydantic.BaseModel):
    score_choice: _const.ScoreChoice = pydantic.Field(
        default=_const.ScoreChoice.VARIANCE_RATIO, validate_default=True
    )
    """scoring choice to determine best cluster composition to use"""

    dist_metric: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.EUCLIDEAN, validate_default=True
    )
    """distance metric to determine best cluster composition to use"""

    min_cluster_size: int = pydantic.Field(default=15, validate_default=True)
    """minimum number of models that must make up a cluster to be considered a non-outlier"""

    num_cluster_bound_lower: int = pydantic.Field(default=8, validate_default=True)
    """minimum number of times cluster bisection will occur on set of pool models"""

    num_cluster_bound_upper: int = pydantic.Field(default=1_500, validate_default=True)
    """maximum number of times cluster bisection will occur on set of pool models"""

    fpca_min_variance_ratio: float = pydantic.Field(default=0.97, validate_default=True)

    seed: int = pydantic.Field(default=42, validate_default=True)
    """seed which allows for reproducibility due the random nature of bisecting k means"""

    agg_type: _const.AggType = pydantic.Field(
        default=_const.AggType.MEDIAN, validate_default=True
    )

    max_non_outlier_cluster_count: int = pydantic.Field(
        default=200, validate_default=True
    )
    """limiting factor on the maximum allowable number of clusters that do not meet the criteria of being classified as an outlier.
    Any result with a count higher than this will be flagged as having a score that is unable to be calculated"""


if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
