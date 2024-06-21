"""
Base settings class for gridmeter
"""
from __future__ import annotations

import pydantic

from enum import Enum
from typing import Any


class BaseSettings(pydantic.BaseModel):
    """Make all property keys case insensitive"""
    @pydantic.model_validator(mode="before")
    def __uppercase_property_keys__(cls, values: Any) -> Any:
        def __upper__(value: Any) -> Any:
            if isinstance(value, dict):
                return {k.upper(): __upper__(v) for k, v in value.items()}
            return value

        return __upper__(values)