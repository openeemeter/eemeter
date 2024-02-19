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
import requests
from pathlib import Path

from eemeter.common.const import TutorialDataChoice

# Define the current directory
current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parents[1] / "data"

# Set download branch
## TODO: THIS BRANCH MUST BE UPDATED TO THE CURRENT BRANCH
branch = "feature/caltrack-2.1-daily"


def load_test_data(data_type: str):
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
        df = _load_file("features.csv")
    
    elif data_type == TutorialDataChoice.SEASONAL_HOUR_DAY_WEEK_LOADSHAPE:
        df = _load_file("seasonal_hourly_day_of_week_loadshape.csv")
    
    elif data_type == TutorialDataChoice.SEASONAL_DAY_WEEK_LOADSHAPE:
        df = _load_file("seasonal_day_of_week_loadshape.csv")
    
    elif data_type == TutorialDataChoice.MONTH_LOADSHAPE:
        df = _load_file("month_loadshape.csv")
    
    elif data_type == TutorialDataChoice.HOURLY_COMPARISON_GROUP_DATA:
        df = pd.concat(
            [
                _load_file("hourly_data_0.parquet"),
                _load_file("hourly_data_1.parquet")
            ], 
            axis=0
        )

    elif data_type == TutorialDataChoice.HOURLY_TREATMENT_DATA:
        df = _load_file("hourly_data_2.parquet")
    
    if data_type not in [TutorialDataChoice.HOURLY_COMPARISON_GROUP_DATA, TutorialDataChoice.HOURLY_TREATMENT_DATA]:
        df = df.set_index("id")
    
    return df


def download_all_test_data():
    """Downloads all the tutorial data to the data directory"""

    # get all repo files
    repo_full_name = 'openeemeter/eemeter'
    path = 'data'

    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"

    if branch != "master":
        url += f"?ref={branch}"

    r = requests.get(url)
    r.raise_for_status()

    files = [file['name'] for file in r.json() if file['type'] == 'file']

    # download all repo files
    for file in files:
        _download_repo_data_file(file)


def _load_file(file):
    attribution_file = data_dir / "attribution.txt"
    file = data_dir / file
    ext = file.suffix

    # if file does not exist, download it
    if not file.exists():
        # always check for attribution file
        if not attribution_file.exists():
           _download_repo_data_file(attribution_file)

        _download_repo_data_file(file)

    if ext == ".csv":
        df = pd.read_csv(file)
    
    elif ext == ".parquet":
        df = pd.read_parquet(file)
    
    else:
        raise ValueError(f"File type {ext} not recognized.")

    return df


def _download_repo_data_file(file):
    repo_full_name = 'openeemeter/eemeter'
    path = 'data'

    url = f"https://raw.githubusercontent.com/{repo_full_name}/{branch}/{path}/{file}"

    r = requests.get(url)
    r.raise_for_status()

    # make directory if it doesn't exist
    if not data_dir.exists():
        data_dir.mkdir()

    with open(data_dir / file, 'wb') as f:
        f.write(r.content)


if __name__ == "__main__":
    df = load_test_data("hourly_treatment_data")
    print(df.index.get_level_values(0).nunique())
    print(df.head())