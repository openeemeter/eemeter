"""
module defines settings to be used for clustering
"""

from __future__ import annotations

import pydantic

import gridmeter._clustering.const as _const
from gridmeter._utils.base_settings import BaseSettings


class Settings(BaseSettings):
    """scoring choice to determine best cluster composition to use"""
    SCORE_CHOICE: _const.ScoreChoice = pydantic.Field(
        default=_const.ScoreChoice.VARIANCE_RATIO, 
        validate_default=True,
    )

    """distance metric to determine best cluster composition to use"""
    DIST_METRIC: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.EUCLIDEAN, 
        validate_default=True,
    )

    """minimum number of models that must make up a cluster to be considered a non-outlier"""
    MIN_CLUSTER_SIZE: int = pydantic.Field(
        default=15, 
        ge=2, 
        validate_default=True,
    )
    
    """minimum number of times cluster bisection will occur on set of pool models"""
    NUM_CLUSTER_BOUND_LOWER: int = pydantic.Field(
        default=8, 
        ge=2, 
        validate_default=True,
    )
    
    """maximum number of times cluster bisection will occur on set of pool models"""
    NUM_CLUSTER_BOUND_UPPER: int = pydantic.Field(
        default=1_500,
        ge=2,
        validate_default=True,
    )
    
    """minimum variance ratio that the fPCA must account for"""
    FPCA_MIN_VARIANCE_RATIO: float = pydantic.Field(
        default=0.97, 
        ge=0.5, 
        le=1.0, 
        validate_default=True)

    """aggregation type for the loadshape"""
    AGG_TYPE: _const.AggType = pydantic.Field(
        default=_const.AggType.MEDIAN, 
        validate_default=True,
    )

    """limiting factor on the maximum allowable number of clusters that do not meet the criteria of being classified as an outlier.
    Any result with a count higher than this will be flagged as having a score that is unable to be calculated"""
    MAX_NON_OUTLIER_CLUSTER_COUNT: int = pydantic.Field(
        default=200,
        ge=2,
        validate_default=True,
    )

    """seed which allows for reproducibility due the random nature of bisecting k means"""
    SEED: int = pydantic.Field(
        default=42, 
        ge=0, 
        validate_default=True,
    )

    @pydantic.model_validator(mode="after")
    def _check_num_cluster_bounds(self):
        if self.num_cluster_bound_lower > self.num_cluster_bound_upper:
            raise ValueError("num_cluster_bound_lower must be less than or equal to num_cluster_bound_upper")

        return self


if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
