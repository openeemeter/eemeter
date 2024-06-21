"""
module defines settings to be used for random sampling
"""

from __future__ import annotations
from typing import Optional

import pydantic

from gridmeter._utils.base_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for random sampling"""
    
    """number meters to randomly sample from comparison pool"""
    N_METERS_TOTAL: Optional[int] = pydantic.Field(
        default=None, 
        validate_default=True,
    )

    """number of meters to randomly sample per treatment"""
    N_METERS_PER_TREATMENT: Optional[int] = pydantic.Field(
        default=4, 
        validate_default=True,
    )

    SEED: Optional[int] = pydantic.Field(
        default=None, 
        validate_default=True,
    )

    """Check if valid settings"""
    @pydantic.model_validator(mode="after")
    def _check_n_meters_choice(self):
        if self.N_METERS_TOTAL is None and self.N_METERS_PER_TREATMENT is None:
            raise ValueError("N_METERS_TOTAL or N_METERS_PER_TREATMENT must be defined")
        
        elif self.N_METERS_TOTAL is not None and self.N_METERS_PER_TREATMENT is not None:
            raise ValueError("N_METERS_TOTAL and N_METERS_PER_TREATMENT cannot be defined together")
        
        elif self.N_METERS_TOTAL is not None and self.N_METERS_TOTAL < 1:
            raise ValueError("N_METERS_TOTAL must be greater than or equal to 1")
        
        elif self.N_METERS_PER_TREATMENT is not None and self.N_METERS_PER_TREATMENT < 1:
            raise ValueError("N_METERS_PER_TREATMENT must be greater than or equal to 1")

        return self
    

if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
