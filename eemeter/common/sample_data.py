#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2023 OpenEEmeter contributors

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

import pandas as pd
from pathlib import Path

from eemeter.common.const import TutorialDataChoice

# Define the current directory
current_dir = Path(__file__).parent
data_dir = current_dir.parents[1] / "data"

def load_data(data_type: str):
    """Returns back tutorial data of the given data type as a dataframe

    Args:
        data_type (str): Must be one of the following:
            - "features"
            - "seasonal_hourly_day_of_week_loadshape"
            - "seasonal_day_of_week_loadshape"
            - "month_loadshape"
            - "hourly_data"

    Returns:
        (dataframe): Returns a dataframe
    """

    # remove all "_" and " " from string and convert to lowercase
    data_type = data_type.lower()
    data_type = data_type.replace("_", "").replace(" ", "")

    valid_list = [k.value for k in TutorialDataChoice]
    keys = [k.lower() for k in TutorialDataChoice.__members__.keys()]
    
    if data_type not in valid_list: 
        raise ValueError(f"Data type {data_type} not recognized. \nMust be one of {keys}.")

    if data_type == TutorialDataChoice.FEATURES:
        df = pd.read_csv(data_dir / "features.csv")
    
    elif data_type == TutorialDataChoice.SEASONAL_HOUR_DAY_WEEK_LOADSHAPE:
        df = pd.read_csv(data_dir / "seasonal_hourly_day_of_week_loadshape.csv")
    
    elif data_type == TutorialDataChoice.SEASONAL_DAY_WEEK_LOADSHAPE:
        df = pd.read_csv(data_dir / "seasonal_day_of_week_loadshape.csv")
    
    elif data_type == TutorialDataChoice.MONTH_LOADSHAPE:
        df = pd.read_csv(data_dir / "month_loadshape.csv")
    
    elif data_type == TutorialDataChoice.HOURLY_DATA:
        df = pd.read_parquet(data_dir / "hourly_data.parquet")
    
    if data_type != TutorialDataChoice.HOURLY_DATA:
        df = df.set_index("id")
    
    return df


if __name__ == "__main__":
    df = load_data("hourly_data")
    print(df.head())