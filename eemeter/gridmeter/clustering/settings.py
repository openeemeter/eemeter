#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from __future__ import annotations
from typing import Optional, Union

import pydantic

import eemeter.gridmeter.clustering.const as _const
from eemeter.gridmeter._utils.base_settings import BaseSettings
from eemeter.common.stats.adaptive_loss import _LOSS_ALPHA_MIN


class Settings(BaseSettings):
    """maximum number of meters to be used in the comparison pool"""
    MAX_COMPARISON_POOL_SIZE: int = pydantic.Field(
        default=10000,
        ge=1,
        validate_default=True,
    )

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

    """how to normalize the loadshape before performing fPCA and clustering"""
    NORMALIZE_METHOD: Optional[_const.NormalizeChoice] = pydantic.Field(
        default=_const.NormalizeChoice.MIN_MAX, 
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

    """treatment meter match loss type"""
    TREATMENT_MATCH_LOSS: Union[str,float] = pydantic.Field(
        default="MAE", 
        validate_default=True,
    )

    """enable/disable multiprocessing"""
    USE_MULTIPROCESSING: bool = pydantic.Field(
        default=True, 
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
        if self.NUM_CLUSTER_BOUND_LOWER >= self.NUM_CLUSTER_BOUND_UPPER:
            raise ValueError("NUM_CLUSTER_BOUND_LOWER must be less than NUM_CLUSTER_BOUND_UPPER")

        return self
    
    @pydantic.model_validator(mode="after")
    def _check_normalize_method(self):
        if self.NORMALIZE_METHOD == "none":
            self.NORMALIZE_METHOD = None

        return self

    """Check if valid settings for treatment meter match loss"""
    @pydantic.model_validator(mode="after")
    def _check_treatment_match_loss(self):
        self._TREATMENT_MATCH_LOSS_ALPHA = self.TREATMENT_MATCH_LOSS

        if isinstance(self._TREATMENT_MATCH_LOSS_ALPHA, str):
            if self._TREATMENT_MATCH_LOSS_ALPHA.upper() in ["SSE", "L2"]:
                self._TREATMENT_MATCH_LOSS_ALPHA = 2.0

            elif self._TREATMENT_MATCH_LOSS_ALPHA.upper() in ["MAE", "L1"]:
                self._TREATMENT_MATCH_LOSS_ALPHA = 1.0
                
            elif self._TREATMENT_MATCH_LOSS_ALPHA != "adaptive":
                raise ValueError("TREATMENT_MATCH_LOSS must be either ['SSE', 'MAE', 'L2', 'L1', 'adaptive'] or float")
            
        else:
            if self._TREATMENT_MATCH_LOSS_ALPHA < _LOSS_ALPHA_MIN:
                raise ValueError(f"TREATMENT_MATCH_LOSS must be greater than {_LOSS_ALPHA_MIN:.0f}")
            
            if self._TREATMENT_MATCH_LOSS_ALPHA > 2:
                raise ValueError("TREATMENT_MATCH_LOSS must be less than 2")

        return self


if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())