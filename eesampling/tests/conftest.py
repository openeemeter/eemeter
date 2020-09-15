#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

   Copyright 2020 EESampling contributors

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
import json
from pkg_resources import resource_stream

import pandas as pd
import pytest
#import numpy as np
#import eesampling

import os 

@pytest.fixture
def monthly_test_data():
   path = os.path.join(os.path.dirname(__file__),'test_data/monthly_test_data.csv')
   return pd.read_csv(path)




