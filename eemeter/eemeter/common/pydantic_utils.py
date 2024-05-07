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

import numpy as np
import pandas as pd
import pydantic

from typing import Any, Optional

class Config:
    arbitrary_types_allowed = True

@pydantic.dataclasses.dataclass(config = Config)
class PydanticDf:
    df: pd.DataFrame

    """list of required column types"""
    column_types: Optional[dict[str, Any]] = None

    @pydantic.model_validator(mode="after")
    def _check_columns(self):
        if self.column_types is not None:
            required_columns = list(self.column_types.keys())
            if not all(column in self.df.columns for column in required_columns):
                raise ValueError(
                    f"Expected columns {required_columns} but got {self.df.columns}"
                )

            for col, col_type in self.column_types.items():
                if col_type is None or col_type is Any:
                    continue

                if self.df[col].dtype != col_type:
                    # attempt to coerce numeric columns
                    if np.issubdtype(col_type, np.number) and np.issubdtype(
                        self.df[col].dtype, np.number
                    ):
                        self.df[col] = self.df[col].astype(col_type)
                    else:
                        raise ValueError(
                            f"Expected column {col} to be of type {col_type} but got {self.df[col].dtype}"
                        )
