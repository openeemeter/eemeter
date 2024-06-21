"""
module defines settings to be used for individual meter matching
"""

from __future__ import annotations

import pydantic

import gridmeter._utils.const as _const
from gridmeter._utils.base_settings import BaseSettings

from enum import Enum
from typing import Optional


class SelectionMethod(str, Enum):
    LEGACY = "legacy"
    MINIMIZE_METER_DISTANCE = "minimize_meter_distance"
    MINIMIZE_LOADSHAPE_DISTANCE = "minimize_loadshape_distance"


class Settings(BaseSettings):
    """Settings for individual meter matching"""

    """distance metric to determine best comparison pool matches"""
    DISTANCE_METRIC: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.EUCLIDEAN, 
        validate_default=True,
    )

    """"""
    SELECTION_METHOD: SelectionMethod = pydantic.Field(
        default=SelectionMethod.LEGACY, 
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
    
    """allow duplicate matches in comparison group"""
    ALLOW_DUPLICATE_MATCHES: bool = pydantic.Field(
        default=False, 
        validate_default=True,
    )
    
    """The maximum distance that a comparison group match can have with a given
       treatment meter. These meters are filtered out after all matching has completed."""
    MAX_DISTANCE_THRESHOLD: Optional[float] = pydantic.Field(
        default=None, 
        validate_default=True,
    )

    """Check if valid settings for treatment meter match loss"""
    @pydantic.model_validator(mode="after")
    def _check_allow_duplicates(self):
        if self.ALLOW_DUPLICATE_MATCHES:
            if (self.SELECTION_METHOD != SelectionMethod.LEGACY) and (self.SELECTION_METHOD != SelectionMethod.MINIMIZE_METER_DISTANCE):
                distance = SelectionMethod.MINIMIZE_METER_DISTANCE.value
                legacy = SelectionMethod.LEGACY.value
                raise ValueError(f"If ALLOW_DUPLICATE_MATCHES is True then SELECTION_METHOD must be '{legacy}' or '{distance}'")

        return self
    

if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
