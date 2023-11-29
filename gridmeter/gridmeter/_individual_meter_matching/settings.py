"""
module defines settings to be used for individual meter matching
"""

from __future__ import annotations

import pydantic

import gridmeter._utils.const as _const
from gridmeter._utils.base_settings import BaseSettings

from typing import Optional


class Settings(BaseSettings):
    """Settings for individual meter matching"""

    """distance metric to determine best comparison pool matches"""
    DISTANCE_METRIC: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.EUCLIDEAN, 
        validate_default=True,
    )
    
    """number of comparison pool matches to each treatment meter"""
    N_MATCHES_PER_TREATMENT: int = pydantic.Field(
        default=4, 
        ge=1, 
        validate_default=True,
    )
    
    """number of treatments to be calculated per chunk to prevent memory issues"""
    N_TREATMENTS_PER_CHUNK: int = pydantic.Field(
        default=10000, 
        ge=1, 
        validate_default=True,
    )
    
    """number of iterations to check for duplicate matches to comparison pool meters and remove them"""
    N_DUPLICATE_CHECK: int = pydantic.Field(
        default=10, 
        ge=0, 
        validate_default=True,
    )
    
    """The maximum distance that a comparison group match can have with a given
       treatment meter. These meters are filtered out after all matching has completed."""
    MAX_DISTANCE_THRESHOLD: Optional[float] = pydantic.Field(
        default=None, 
        validate_default=True,
    )
    

if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
