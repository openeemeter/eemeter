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

import pydantic

from enum import Enum
from typing import Any


class BaseSettings(pydantic.BaseModel):
    class Config:
        frozen = True

    """Make all property keys case insensitive"""

    @pydantic.model_validator(mode="before")
    def __uppercase_property_keys__(cls, values: Any) -> Any:
        def __upper__(value: Any) -> Any:
            if isinstance(value, dict):
                return {k.upper() if isinstance(k, str) else k: __upper__(v) for k, v in value.items()}
            return value

        return __upper__(values)
