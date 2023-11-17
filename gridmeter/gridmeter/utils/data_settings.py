"""
module defines settings to be used for individual meter matching
"""

from __future__ import annotations

import pydantic

import gridmeter.utils.const as _const

from typing import Optional


class Settings(pydantic.BaseModel):
    distance_metric: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.EUCLIDEAN, validate_default=True
    )
    """distance metric to determine best comparison pool matches"""

    n_matches_per_treatment: int = pydantic.Field(default=4, validate_default=True)
    """number of comparison pool matches to each treatment meter"""

    n_treatments_per_chunk: int = pydantic.Field(default=10000, validate_default=True)
    """number of treatments to be calculated per chunk to prevent memory issues"""

    n_duplicate_check: int = pydantic.Field(default=10, validate_default=True)
    """number of iterations to check for duplicate matches to comparison pool meters and remove them"""

    max_distance_threshold: Optional[float] = pydantic.Field(
        default=None, validate_default=True
    )
    """The maximum distance that a comparison group match can have with a given
       treatment meter. These meters are filtered out after all matching has completed."""


if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
