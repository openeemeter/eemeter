#!/usr/bin/env python3
# -*_ coding: utf-8 -*- 

"""

Copyright 2020 EESampling contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may opbtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS 
WITHOUT WARRANTIES OR CONDITOINS OFANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 
"""

import pytest

from eesampling import RandomSamplingModel

def test_random_sampling(monthly_test_data):
    df = monthly_test_data
    m = RandomSamplingModel()
    result = m.sample(X_pool=df, n_outputs=1000)
    assert len(result) == 1000

    with pytest.raises(ValueError) as e:
        m.sample(X_pool=0, n_outputs=10)
    assert str(e.value) == "X_pool must be a pandas DataFrame with one row per meter."

    with pytest.raises(ValueError) as e:
        m.sample(X_pool=df.head(10), n_outputs=11)
    assert str(e.value) == "11 outputs requested, but only 10 available in pool."


    m = RandomSamplingModel(random_seed = None)
    r1 = m.sample(X_pool=df, n_outputs=100)
    r2 = m.sample(X_pool=df, n_outputs=100)
    assert not r1.equals(r2)

    m = RandomSamplingModel(random_seed = 3)
    r1 = m.sample(X_pool=df, n_outputs=100)
    r2 = m.sample(X_pool=df, n_outputs=100)
    assert r1.equals(r2)



