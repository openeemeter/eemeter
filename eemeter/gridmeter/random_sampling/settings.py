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
from typing import Optional

import pydantic

from eemeter.gridmeter._utils.base_settings import BaseSettings


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
