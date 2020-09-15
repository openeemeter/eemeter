#!/usr/bin/env python3

import pandas as pd
import numpy as np 

n_test_meters = 10000
monthly_range = np.array([0.3, 0.4, 0.4, 0.6, 0.8, 0.9, 1.0, 0.9, 0.7, 0.6, 0.5, 0.3])
monthly_test_data = pd.concat([pd.DataFrame({'id': f'meter_{i}', 
                  'data': monthly_range*np.random.random()}) for i in range(n_test_meters)])

monthly_test_data.to_csv('monthly_test_data.csv')
